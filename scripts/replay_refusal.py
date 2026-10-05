#!/usr/bin/env python3
"""Replay a refusal dump through litellm to bisect what triggers the refusal.

Usage:
    python scripts/replay_refusal.py <dump.json> [--alt-model <model>]

The script replays the dump through litellm.completion with the same model
and kwargs and prints finish_reason and usage for each variant:
    - baseline: the dump as-is, to confirm it refuses reproducibly
    - no_reasoning: drop reasoning_effort
    - no_security: system prompt with "B. SECURITY VULNERABILITIES" block removed
    - no_repl: system prompt with RLM REPL template paragraph removed
    - no_patch: user message with changed_files[*].patch replaced by placeholder
    - alt_model: same messages on a model given with --alt-model
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import litellm


def _strip_api_keys(kwargs: dict) -> dict:
    """Remove API keys from kwargs for safe display."""
    safe = {}
    for key, value in kwargs.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            safe[key] = "<redacted>"
        else:
            safe[key] = value
    return safe


def _run_completion(model: str, messages: list[dict], lm_kwargs: dict) -> dict[str, Any]:
    """Run litellm.completion and return result info."""
    try:
        # Filter out kwargs that litellm doesn't accept directly
        safe_kwargs = {k: v for k, v in lm_kwargs.items()
                       if k not in ("cache_control_injection_points",)}
        response = litellm.completion(
            model=model,
            messages=messages,
            **safe_kwargs,
        )
        choice = response.choices[0] if response.choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else getattr(choice, "message", {})
        usage = response.usage
        return {
            "success": True,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", "unknown"),
            "content": message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""),
            "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
            "completion_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _make_variant_baseline(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Baseline: use messages and kwargs as-is."""
    return messages, lm_kwargs


def _make_variant_no_reasoning(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove reasoning_effort from kwargs."""
    new_kwargs = {k: v for k, v in lm_kwargs.items() if k != "reasoning_effort"}
    return messages, new_kwargs


def _remove_security_block(content: str) -> str:
    """Remove the 'B. SECURITY VULNERABILITIES' section from system prompt."""
    # Match from "B. SECURITY VULNERABILITIES" to next "[A-Z]. " or end
    pattern = r"B\.\s*SECURITY\s*VULNERABILITIES.*?(?=\n[A-Z]\.\s|\Z)"
    return re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)


def _make_variant_no_security(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove security vulnerabilities section from system prompt."""
    new_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _remove_security_block(content)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _remove_repl_paragraph(content: str) -> str:
    """Remove the RLM REPL template paragraph from system prompt."""
    # Match "You have access to a Python REPL" paragraph
    pattern = r"You have access to a Python REPL.*?call SUBMIT\(\)"
    return re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)


def _make_variant_no_repl(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove REPL/code-execution framing from system prompt."""
    new_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _remove_repl_paragraph(content)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _remove_patches(content: str) -> str:
    """Replace patch content with placeholder."""
    # Replace ```diff blocks with placeholder
    pattern = r"```diff.*?```"
    return re.sub(pattern, "[PATCH CONTENT REDACTED]", content, flags=re.DOTALL)


def _make_variant_no_patch(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Replace patch content in user message with placeholder."""
    new_messages = []
    for msg in messages:
        if msg.get("role") in ("user", "human"):
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _remove_patches(content)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _print_result(variant: str, result: dict[str, Any]) -> None:
    """Print the result of a variant run."""
    print(f"\n{'='*60}")
    print(f"Variant: {variant}")
    print(f"{'='*60}")
    if result["success"]:
        print(f"  finish_reason: {result['finish_reason']}")
        print(f"  prompt_tokens: {result['prompt_tokens']}")
        print(f"  completion_tokens: {result['completion_tokens']}")
        print(f"  reasoning_tokens: {result['reasoning_tokens']}")
        content_preview = result['content'][:200] if result['content'] else "<empty>"
        print(f"  content_preview: {content_preview}...")
    else:
        print(f"  ERROR: {result['error']}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Replay a refusal dump to bisect what triggers the refusal."
    )
    parser.add_argument("dump", type=Path, help="Path to the refusal dump JSON file")
    parser.add_argument(
        "--alt-model",
        help="Alternative model to test (e.g., bedrock/anthropic.claude-sonnet-4.1)",
    )
    args = parser.parse_args()

    dump_path = args.dump
    if not dump_path.exists():
        print(f"Error: Dump file not found: {dump_path}", file=sys.stderr)
        return 1

    # Load dump
    with open(dump_path, "r", encoding="utf-8") as f:
        dump = json.load(f)

    model = dump.get("model", "unknown")
    lm_kwargs = dump.get("lm_kwargs", {})
    messages = dump.get("messages", [])

    print(f"Loaded refusal dump: {dump_path}")
    print(f"  Model: {model}")
    print(f"  Messages: {len(messages)}")
    print(f"  LM kwargs: {_strip_api_keys(lm_kwargs)}")

    # Run variants
    variants = [
        ("baseline", _make_variant_baseline),
        ("no_reasoning", _make_variant_no_reasoning),
        ("no_security", _make_variant_no_security),
        ("no_repl", _make_variant_no_repl),
        ("no_patch", _make_variant_no_patch),
    ]

    for variant_name, variant_fn in variants:
        variant_messages, variant_kwargs = variant_fn(messages, lm_kwargs)
        result = _run_completion(model, variant_messages, variant_kwargs)
        _print_result(variant_name, result)

    # Run alt_model variant if specified
    if args.alt_model:
        result = _run_completion(args.alt_model, messages, lm_kwargs)
        _print_result(f"alt_model ({args.alt_model})", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

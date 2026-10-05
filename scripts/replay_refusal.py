#!/usr/bin/env python3
"""Replay a refusal dump through litellm to bisect what triggers the refusal.

Usage:
    python scripts/replay_refusal.py <dump.json> [--alt-model <model>] [--raw] [--only v1,v2]

The script replays the dump through litellm.completion with the same model
and kwargs and prints finish_reason and usage for each variant:
    - baseline: the dump as-is, to confirm it refuses reproducibly
    - no_reasoning: drop reasoning_effort
    - no_security: system prompt with "B. SECURITY VULNERABILITIES" block removed
    - no_repl: system prompt with RLM REPL template paragraph removed
    - no_tools: system prompt with tool docs removed
    - no_security_no_repl: both security and REPL removed
    - no_patch: user message with changed_files[*].patch replaced by placeholder
    - neutral_user: replace user message with neutral content
    - no_cache_control: drop cache_control_injection_points
    - trivial: minimal system "You are a helpful assistant.", user "Reply with OK."
    - core_only: no_security + no_repl + neutral_user combined
    - no_sig: delete from "Specific instructions:" up to "You are tasked with producing"
    - sig_A_only: keep only section A (BUGS & LOGIC ERRORS)
    - sig_C_only: keep only section C (CODE SMELLS)
    - core_no_preamble / core_no_intro / core_no_head: core_only minus the RLM
      preamble, the signature intro, or both
    - head_only / preamble_only / intro_only: system = only that head part, neutral user
    - preamble_no_head / preamble_no_layout: preamble_only minus one stock sentence
    - preamble_rewrite(_only): preamble with stock sentences neutralised (did not help)
    - pre_drop_<field>: preamble_only minus one RLM field line
      (variables_info, repl_history, iteration, reasoning, code)
    - pre_no_fields: preamble_only minus all five field lines
    - full_reasoning_desc / full_code_desc: full request with that field's desc neutralised
    Note: no_patch is a no-op on dumps whose user message only holds a truncated
    variables_info preview (no patch text present).
    - alt_model: same messages on a model given with --alt-model
"""

import argparse
import json
import os
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


def _messages_changed(orig: list[dict], new: list[dict]) -> bool:
    """Check if messages list differs from original."""
    return orig != new


def _kwargs_changed(orig: dict, new: dict) -> bool:
    """Check if kwargs dict differs from original."""
    return orig != new


def _run_completion(model: str, messages: list[dict], lm_kwargs: dict) -> dict[str, Any]:
    """Run litellm.completion and return result info."""
    try:
        # Pass lm_kwargs through as-is (no filter)
        response = litellm.completion(
            model=model,
            messages=messages,
            **lm_kwargs,
        )
        choice = response.choices[0] if response.choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else getattr(choice, "message", {})
        usage = response.usage

        # Extract reasoning_tokens from completion_tokens_details if available
        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)

        return {
            "success": True,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", "unknown"),
            "content": message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""),
            "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
            "completion_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
            "reasoning_tokens": reasoning_tokens,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _run_raw_bedrock(model: str, messages: list[dict], lm_kwargs: dict) -> dict[str, Any]:
    """Run boto3 bedrock-runtime converse directly and return result info."""
    try:
        import boto3
    except ImportError:
        return {"success": False, "error": "boto3 not installed"}

    try:
        # Strip bedrock/converse/ prefix from model
        model_id = model.replace("bedrock/converse/", "")

        # Build system and messages for converse API
        system_text = ""
        converse_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_text = content
            elif msg.get("role") in ("user", "human"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    converse_messages.append({
                        "role": "user",
                        "content": [{"text": content}]
                    })
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    converse_messages.append({
                        "role": "assistant",
                        "content": [{"text": content}]
                    })

        # Build inference config from lm_kwargs
        inference_config = {}
        if "max_tokens" in lm_kwargs:
            inference_config["maxTokens"] = lm_kwargs["max_tokens"]
        if "temperature" in lm_kwargs:
            inference_config["temperature"] = lm_kwargs["temperature"]

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("bedrock-runtime", region_name=region)

        request_kwargs = {
            "modelId": model_id,
            "messages": converse_messages,
        }
        if system_text:
            request_kwargs["system"] = [{"text": system_text}]
        if inference_config:
            request_kwargs["inferenceConfig"] = inference_config

        response = client.converse(**request_kwargs)

        # Extract response details
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        content_text = ""
        for block in content_blocks:
            if "text" in block:
                content_text += block["text"]

        usage = response.get("usage", {})
        stop_reason = output.get("stopReason", "unknown")
        request_id = response.get("ResponseMetadata", {}).get("RequestId", "unknown")

        return {
            "success": True,
            "finish_reason": stop_reason,
            "content": content_text,
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "reasoning_tokens": 0,
            "raw_stop_reason": stop_reason,
            "request_id": request_id,
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
    """Remove the 'B. SECURITY VULNERABILITIES' section from system prompt.

    Delete from '─── B. SECURITY' up to (not including) '─── C. CODE SMELLS'
    """
    # Match from "─── B. SECURITY" up to but not including "─── C. CODE SMELLS"
    pattern = r"(─── B\.\s*SECURITY.*?)(?=─── C\.\s*CODE SMELLS)"
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
    """Remove the RLM REPL template from system prompt.

    Delete from "You have access to a Python REPL" to the end of the system message.
    """
    pattern = r"You have access to a Python REPL.*"
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


def _remove_tools_section(content: str) -> str:
    """Remove the tool documentation from system prompt.

    Delete from "Additional tools available" to the end of the system message.
    """
    pattern = r"Additional tools available.*"
    return re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)


def _make_variant_no_tools(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove tool docs from system prompt."""
    new_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _remove_tools_section(content)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _make_variant_no_security_no_repl(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove both security section and REPL template from system prompt."""
    # First apply no_security
    msgs1, kwargs1 = _make_variant_no_security(messages, lm_kwargs)
    # Then apply no_repl
    return _make_variant_no_repl(msgs1, kwargs1)


def _remove_patches(content: str) -> str:
    """Replace patch content in JSON strings with placeholder."""
    # Handle escaped newlines in JSON strings
    # The patch field contains actual newlines that are escaped in JSON
    # Match "patch": "..." with content containing escaped newlines
    pattern = r'("patch"\s*:\s*")((?:[^"\\]|\\.)*?)(")'

    def replace_patch(match):
        return match.group(1) + "[PATCH REDACTED]" + match.group(3)

    return re.sub(pattern, replace_patch, content, flags=re.DOTALL)


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


def _make_variant_neutral_user(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Replace user message with neutral content."""
    new_messages = []
    for msg in messages:
        if msg.get("role") in ("user", "human"):
            new_msg = dict(msg)
            new_msg["content"] = "variables_info: []\n\nrepl_history: entries=[] max_output_chars=10000\n\niteration: 1/5"
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _make_variant_no_cache_control(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove cache_control_injection_points from kwargs."""
    new_kwargs = {k: v for k, v in lm_kwargs.items() if k != "cache_control_injection_points"}
    return messages, new_kwargs


def _make_variant_trivial(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Minimal system and user messages to test if model accepts anything."""
    new_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            new_msg = dict(msg)
            new_msg["content"] = "You are a helpful assistant."
            new_messages.append(new_msg)
        elif msg.get("role") in ("user", "human"):
            new_msg = dict(msg)
            new_msg["content"] = "Reply with OK."
            new_messages.append(new_msg)
        else:
            new_messages.append(dict(msg))
    # Keep lm_kwargs as-is
    return new_messages, lm_kwargs


def _make_variant_core_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Combine no_security + no_repl + neutral_user."""
    # Apply no_security
    msgs1, kwargs1 = _make_variant_no_security(messages, lm_kwargs)
    # Apply no_repl
    msgs2, kwargs2 = _make_variant_no_repl(msgs1, kwargs1)
    # Apply neutral_user
    return _make_variant_neutral_user(msgs2, kwargs2)


def _remove_signature_instructions(content: str) -> str:
    """Remove from 'Specific instructions:' up to 'You are tasked with producing'.

    Keeps the header, schema and REPL, drops the whole CodeReviewSignature docstring.
    """
    # Match from "Specific instructions:" up to but not including "You are tasked with producing"
    pattern = r"(Specific instructions:.*?)(?=You are tasked with producing)"
    return re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)


def _make_variant_no_sig(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Remove the CodeReviewSignature docstring from system prompt."""
    new_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _remove_signature_instructions(content)
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, lm_kwargs


def _keep_only_section(content: str, section_letter: str) -> str:
    """Keep only the specified section (A or C) from the signature.

    Section A: BUGS & LOGIC ERRORS
    Section C: CODE SMELLS
    """
    # Find the ANALYZE CHANGES header and OUTPUT RULES footer
    header_match = re.search(r"(ANALYZE CHANGES.*?)(?=─── A\.)", content, re.DOTALL | re.IGNORECASE)
    footer_match = re.search(r"(OUTPUT RULES.*?)$", content, re.DOTALL | re.IGNORECASE)

    header = header_match.group(1) if header_match else "ANALYZE CHANGES\n"
    footer = footer_match.group(1) if footer_match else ""

    if section_letter.upper() == "A":
        # Keep section A (from ─── A. to ─── B.)
        section_match = re.search(r"(─── A\.\s*BUGS.*?)(?=─── B\.\s*SECURITY)", content, re.DOTALL | re.IGNORECASE)
    elif section_letter.upper() == "C":
        # Keep section C (from ─── C. to OUTPUT RULES)
        section_match = re.search(r"(─── C\.\s*CODE SMELLS.*?)(?=OUTPUT RULES)", content, re.DOTALL | re.IGNORECASE)
    else:
        section_match = None

    section = section_match.group(1) if section_match else ""

    # Combine header + section + footer
    return header + section + footer


def _make_variant_sig_A_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Keep only section A (BUGS & LOGIC ERRORS) from system prompt.

    Run on top of core_only.
    """
    # First apply core_only
    msgs1, kwargs1 = _make_variant_core_only(messages, lm_kwargs)

    new_messages = []
    for msg in msgs1:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _keep_only_section(content, "A")
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, kwargs1


def _make_variant_sig_C_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Keep only section C (CODE SMELLS) from system prompt.

    Run on top of core_only.
    """
    # First apply core_only
    msgs1, kwargs1 = _make_variant_core_only(messages, lm_kwargs)

    new_messages = []
    for msg in msgs1:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = _keep_only_section(content, "C")
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages, kwargs1


_PREAMBLE_END = "Specific instructions:"
_INTRO_END = "═══"  # first banner line, right before ANALYZE CHANGES


def _split_head(content: str) -> tuple[str, str, str]:
    """Split system content into (rlm_preamble, sig_intro, rest).

    rlm_preamble: start .. "Specific instructions:" (DSPy RLM field listing)
    sig_intro:    "Specific instructions: Detect VERIFIED ..." .. first ═══ banner
    rest:         ANALYZE CHANGES banner onward
    """
    p = content.find(_PREAMBLE_END)
    i = content.find(_INTRO_END, p if p >= 0 else 0)
    if p < 0 or i < 0:
        return "", "", content
    return content[:p], content[p:i], content[i:]


def _map_system(messages: list[dict], fn) -> list[dict]:
    """Apply fn to the system message content."""
    out = []
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            out.append({**msg, "content": fn(msg["content"])})
        else:
            out.append(msg)
    return out


def _make_variant_core_no_preamble(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """core_only minus the RLM preamble (keeps sig intro)."""
    msgs, kw = _make_variant_core_only(messages, lm_kwargs)
    return _map_system(msgs, lambda c: "".join(_split_head(c)[1:])), kw


def _make_variant_core_no_intro(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """core_only minus the signature intro (keeps RLM preamble)."""
    def fn(c: str) -> str:
        pre, _, rest = _split_head(c)
        return pre + rest
    msgs, kw = _make_variant_core_only(messages, lm_kwargs)
    return _map_system(msgs, fn), kw


def _make_variant_core_no_head(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """core_only minus preamble and intro: tests A+C together."""
    msgs, kw = _make_variant_core_only(messages, lm_kwargs)
    return _map_system(msgs, lambda c: _split_head(c)[2]), kw


def _make_variant_head_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """System = preamble + intro only, neutral user."""
    msgs, kw = _make_variant_neutral_user(messages, lm_kwargs)
    return _map_system(msgs, lambda c: "".join(_split_head(c)[:2])), kw


def _make_variant_preamble_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """System = RLM preamble only, neutral user."""
    msgs, kw = _make_variant_neutral_user(messages, lm_kwargs)
    return _map_system(msgs, lambda c: _split_head(c)[0]), kw


def _make_variant_intro_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """System = signature intro only, neutral user."""
    msgs, kw = _make_variant_neutral_user(messages, lm_kwargs)
    return _map_system(msgs, lambda c: _split_head(c)[1]), kw


# Stock TwoStepAdapter preamble sentences (dspy 3.4 format_task_description)
_STOCK_HEAD = "You are a helpful assistant that can solve tasks based on user input.\n"
_STOCK_INPUTS = "As input, you will be provided with:\n"
_STOCK_OUTPUTS = "Your outputs must contain:\n"
_STOCK_LAYOUT = "You should lay out your outputs in detail so that your answer can be understood by another agent\n"


def _rewrite_preamble(c: str) -> str:
    """Mirror DiagnosticTwoStepAdapter.format_task_description."""
    return (
        c.replace(_STOCK_HEAD, "Complete the task below.\n", 1)
        .replace(_STOCK_INPUTS, "Inputs:\n", 1)
        .replace(_STOCK_OUTPUTS, "Outputs:\n", 1)
        .replace(_STOCK_LAYOUT, "", 1)
    )


def _make_variant_preamble_rewrite(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Full request with the preamble as codespy's adapter now renders it."""
    return _map_system(messages, _rewrite_preamble), lm_kwargs


def _make_variant_preamble_rewrite_only(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """System = rewritten preamble only, neutral user."""
    msgs, kw = _make_variant_preamble_only(messages, lm_kwargs)
    return _map_system(msgs, _rewrite_preamble), kw


def _drop(s: str):
    return lambda c: c.replace(s, "", 1)


def _make_variant_preamble_no_head(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """preamble_only minus the 'helpful assistant' sentence."""
    msgs, kw = _make_variant_preamble_only(messages, lm_kwargs)
    return _map_system(msgs, _drop(_STOCK_HEAD)), kw


def _make_variant_preamble_no_layout(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """preamble_only minus the 'understood by another agent' sentence."""
    msgs, kw = _make_variant_preamble_only(messages, lm_kwargs)
    return _map_system(msgs, _drop(_STOCK_LAYOUT.rstrip("\n"))), kw


# RLM action-signature field lines (dspy/predict/rlm.py), as rendered in the preamble
_RLM_FIELDS = ("variables_info", "repl_history", "iteration", "reasoning", "code")


def _drop_field_lines(c: str, fields: tuple[str, ...]) -> str:
    """Drop numbered '`field` (type): desc' lines from the preamble."""
    for f in fields:
        c = re.sub(rf"^\d+\. `{f}` \([^)]*\): .*\n", "", c, count=1, flags=re.M)
    return c


def _pre_drop(field: str):
    def fn(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
        msgs, kw = _make_variant_preamble_only(messages, lm_kwargs)
        return _map_system(msgs, lambda c: _drop_field_lines(c, (field,))), kw
    fn.__doc__ = f"preamble_only minus the `{field}` line."
    return fn


def _make_variant_pre_no_fields(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """preamble_only minus all five field lines (headers + layout sentence only)."""
    msgs, kw = _make_variant_preamble_only(messages, lm_kwargs)
    return _map_system(msgs, lambda c: _drop_field_lines(c, _RLM_FIELDS)), kw


_REASONING_DESC = "Think step-by-step: what do you know? What remains? Plan your next action."
_CODE_DESC = "Python code to execute."


def _make_variant_full_reasoning_desc(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Full request with the `reasoning` field description neutralised."""
    return _map_system(messages, lambda c: c.replace(_REASONING_DESC, "Short plan for the next step.")), lm_kwargs


def _make_variant_full_code_desc(messages: list[dict], lm_kwargs: dict) -> tuple[list[dict], dict]:
    """Full request with the `code` field description neutralised."""
    return _map_system(messages, lambda c: c.replace(_CODE_DESC, "Python snippet for the next step.")), lm_kwargs


def _print_result(
    variant: str,
    result: dict[str, Any],
    orig_messages: list[dict],
    variant_messages: list[dict],
    orig_kwargs: dict,
    variant_kwargs: dict,
) -> None:
    """Print the result of a variant run."""
    msg_changed = _messages_changed(orig_messages, variant_messages)
    kwargs_changed = _kwargs_changed(orig_kwargs, variant_kwargs)
    changed = msg_changed or kwargs_changed

    print(f"\n{'='*60}")
    print(f"Variant: {variant}")
    print(f"changed={changed} (msg_changed={msg_changed}, kwargs_changed={kwargs_changed})")
    print(f"{'='*60}")
    if result["success"]:
        print(f"  finish_reason: {result['finish_reason']}")
        print(f"  prompt_tokens: {result['prompt_tokens']}")
        print(f"  completion_tokens: {result['completion_tokens']}")
        print(f"  reasoning_tokens: {result['reasoning_tokens']}")
        if "raw_stop_reason" in result:
            print(f"  raw_stop_reason: {result['raw_stop_reason']}")
        if "request_id" in result:
            print(f"  request_id: {result['request_id']}")
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
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Run baseline through boto3 bedrock-runtime directly and print raw response",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated list of variants to run (e.g., 'baseline,trivial,core_only')",
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

    # Handle --raw: run boto3 directly for baseline
    if args.raw:
        print(f"\n{'='*60}")
        print(f"RAW BEDROCK CONVERSE (baseline)")
        print(f"{'='*60}")
        result = _run_raw_bedrock(model, messages, lm_kwargs)
        if result["success"]:
            print(f"  stopReason: {result.get('raw_stop_reason', result['finish_reason'])}")
            print(f"  prompt_tokens: {result['prompt_tokens']}")
            print(f"  completion_tokens: {result['completion_tokens']}")
            print(f"  request_id: {result.get('request_id', 'N/A')}")
            content_preview = result['content'][:200] if result['content'] else "<empty>"
            print(f"  content_preview: {content_preview}...")
        else:
            print(f"  ERROR: {result['error']}")

    # Turn off litellm caching
    litellm.cache = None

    # Define all variants
    all_variants = [
        ("baseline", _make_variant_baseline),
        ("no_reasoning", _make_variant_no_reasoning),
        ("no_security", _make_variant_no_security),
        ("no_repl", _make_variant_no_repl),
        ("no_tools", _make_variant_no_tools),
        ("no_security_no_repl", _make_variant_no_security_no_repl),
        ("no_patch", _make_variant_no_patch),
        ("neutral_user", _make_variant_neutral_user),
        ("no_cache_control", _make_variant_no_cache_control),
        ("trivial", _make_variant_trivial),
        ("core_only", _make_variant_core_only),
        ("no_sig", _make_variant_no_sig),
        ("sig_A_only", _make_variant_sig_A_only),
        ("sig_C_only", _make_variant_sig_C_only),
        ("core_no_preamble", _make_variant_core_no_preamble),
        ("core_no_intro", _make_variant_core_no_intro),
        ("core_no_head", _make_variant_core_no_head),
        ("head_only", _make_variant_head_only),
        ("preamble_only", _make_variant_preamble_only),
        ("intro_only", _make_variant_intro_only),
        ("preamble_no_head", _make_variant_preamble_no_head),
        ("preamble_no_layout", _make_variant_preamble_no_layout),
        ("preamble_rewrite_only", _make_variant_preamble_rewrite_only),
        ("preamble_rewrite", _make_variant_preamble_rewrite),
        *((f"pre_drop_{f}", _pre_drop(f)) for f in _RLM_FIELDS),
        ("pre_no_fields", _make_variant_pre_no_fields),
        ("full_reasoning_desc", _make_variant_full_reasoning_desc),
        ("full_code_desc", _make_variant_full_code_desc),
    ]

    # Filter variants if --only specified
    if args.only:
        only_set = set(args.only.split(","))
        variants = [(n, f) for n, f in all_variants if n in only_set]
        if not variants:
            print(f"Error: No variants matched '{args.only}'", file=sys.stderr)
            return 1
    else:
        variants = all_variants

    for variant_name, variant_fn in variants:
        variant_messages, variant_kwargs = variant_fn(messages, lm_kwargs)
        result = _run_completion(model, variant_messages, variant_kwargs)
        _print_result(variant_name, result, messages, variant_messages, lm_kwargs, variant_kwargs)

    # Run alt_model variant if specified
    if args.alt_model:
        result = _run_completion(args.alt_model, messages, lm_kwargs)
        _print_result(f"alt_model ({args.alt_model})", result, messages, messages, lm_kwargs, lm_kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(main())

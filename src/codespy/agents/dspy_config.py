"""DSPy and LiteLLM configuration utilities."""

import logging

import dspy  # type: ignore[import-untyped]
from dspy.adapters.two_step_adapter import TwoStepAdapter  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]

from codespy.config import Settings, get_settings
from codespy.config_memory import LLMSettings, REFLECTION_MODULES


logger = logging.getLogger(__name__)


def _resolve_max_tokens(model: str, max_tokens: int) -> int:
    """Clamp a configured output budget to the model's real output ceiling.

    ``max_tokens`` is an *output* budget, not a context window, and providers
    reject a request that asks for more than the model can emit (Anthropic
    returns a 400). Clamping lets a single generous default be configured
    globally and still resolve to a valid value per model.

    Models LiteLLM doesn't know about (Ollama, custom endpoints) have no
    published ceiling, so the configured value is passed through unchanged.

    Args:
        model: The LiteLLM model identifier.
        max_tokens: The configured output token budget.

    Returns:
        The budget, reduced to the model's ceiling when one is known.
    """
    try:
        ceiling = litellm.get_max_tokens(model)
    except Exception:  # Unmapped model — no published ceiling to clamp to.
        return max_tokens
    if not ceiling:
        return max_tokens
    return min(max_tokens, ceiling)


def _supports_cache_control(model: str) -> bool:
    """Whether the model uses explicit Anthropic-style cache_control markers.

    Returns True only for models where the provider expects ``cache_control``
    fields in messages to enable prompt caching. Models with automatic caching
    (OpenAI prefix-match, Gemini) or no caching return False — sending markers
    to them is either pointless or causes provider errors.

    Detection: ``cache_creation_input_token_cost`` is non-None in LiteLLM's
    model database only for Anthropic-style providers that charge separately
    for cache writes, which correlates exactly with explicit marker support.

    Falls back to provider-prefix heuristic when model info is unavailable.
    """
    try:
        info = litellm.get_model_info(model)
        if info.get("cache_creation_input_token_cost") is not None:
            return True
        return False
    except Exception:
        # Model not in LiteLLM DB (Ollama offline, custom endpoint).
        # Fall back to prefix heuristic.
        lower = model.lower()
        if lower.startswith("anthropic/"):
            return True
        if lower.startswith("bedrock/") and "anthropic" in lower:
            return True
        return False


def new_lm(settings: Settings, config: LLMSettings) -> dspy.LM:
    """Build a ``dspy.LM`` for resolved LLM settings.

    The single place LMs are constructed, so the cross-cutting concerns
    (timeout, retries, output budget, provider-side prompt caching) are applied
    uniformly.

    ``reasoning_effort`` is always forwarded to LiteLLM through ``dspy.LM``'s
    ``**kwargs``; LiteLLM maps it onto each provider's native parameter
    (Anthropic ``thinking.budget_tokens``, OpenAI ``reasoning_effort``,
    Ollama ``think``, ...). ``drop_params=True`` ensures that models/providers
    which do not support it gracefully ignore the parameter instead of crashing.

    ``max_tokens`` must be passed explicitly: omitting it makes LiteLLM fall
    back to its own 4096 default, which silently truncates responses (DSPy then
    warns about ``max_tokens=None``). Reasoning tokens are charged against this
    budget, so it is clamped to — not capped below — the model's ceiling.

    Args:
        settings: Application settings (timeout / retries / prompt caching).
        config: Resolved settings from ``Settings.get_llm_config()``.

    Returns:
        A configured ``dspy.LM``.
    """
    lm_kwargs: dict = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": _resolve_max_tokens(config.model, config.max_tokens),
        "timeout": settings.llm_timeout,
        "num_retries": settings.llm_retries,
        "drop_params": True,
        "reasoning_effort": config.reasoning_effort,
    }
    # Cache system prompts via explicit Anthropic-style cache_control markers.
    # Only injected for providers that use explicit markers (Anthropic, Bedrock
    # Anthropic); OpenAI/Gemini have automatic caching that needs no markers.
    if settings.enable_prompt_caching and _supports_cache_control(config.model):
        lm_kwargs["cache_control_injection_points"] = [
            {"location": "message", "role": "system"}
        ]
    elif settings.enable_prompt_caching:
        logger.debug(
            f"Prompt caching enabled but model {config.model} does not use "
            f"explicit cache_control markers — skipping injection"
        )
    return dspy.LM(**lm_kwargs)


def lm_context(name: str):
    """Return a ``dspy.context`` applying the LM configured for ``name``.

    ``name`` is a signature or reflection module name — see
    ``Settings.get_llm_config``. Enter this around the predictor call so the
    named unit of work runs on its own configured model, falling back to the
    top-level defaults when it declares no overrides.

    Args:
        name: The signature or reflection module name.

    Returns:
        A context manager that scopes the LM to the enclosed block.
    """
    settings = get_settings()
    llm_config = settings.get_llm_config(name)
    lm = new_lm(settings, llm_config)
    return dspy.context(lm=lm)



def configure_dspy(settings: Settings) -> None:

    """Configure DSPy with the LLM backend for reliable structured output.

    This configures DSPy with:
    - TwoStepAdapter for robust structured output parsing:
      * Stage 1: Main LM generates free-form reasoning without format constraints
      * Stage 2: Extraction LM extracts structured fields from free-form response
    - Global timeout and retries for reliability
    - Provider-side prompt caching (when enabled)
    - Memory caching for LLM responses

    TwoStepAdapter decouples reasoning quality from format compliance,
    solving ChatAdapter parsing failures with ReAct agents.

    Args:
        settings: Application settings containing model and API key configuration.
    """
    model = settings.default_model

    # Configure LiteLLM environment if needed
    if settings.openai_api_key:
        litellm.openai_key = settings.openai_api_key
    if settings.anthropic_api_key:
        litellm.anthropic_key = settings.anthropic_api_key
    # Set up AWS credentials for Bedrock if using Bedrock model
    if model.startswith("bedrock/"):
        import os
        os.environ["AWS_REGION_NAME"] = settings.aws_region
        if settings.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key

    # Global fallback LM, used by any predictor not wrapped in lm_context().
    # An unknown name resolves purely from the top-level defaults.
    defaults = settings.get_llm_config("default")
    lm = new_lm(settings, defaults)

    # Extraction LM for TwoStepAdapter's second stage: deterministic field extraction
    # from the main LM's free-form response. Never uses reasoning; temperature=0.0 for
    # deterministic output.
    extraction_model = defaults.extraction_model
    extraction_lm = new_lm(
        settings, defaults.model_copy(update={
            "model": extraction_model,
            "reasoning_effort": None,
            "temperature": 0.0,
        })
    )


    dspy.settings.configure(
        lm=lm,
        adapter=TwoStepAdapter(extraction_lm),  # TwoStepAdapter solves ChatAdapter parsing failures
    )

    # Enable memory-only caching for LLM calls (no disk caching)
    dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=False, memory_max_entries=10000)

    if not settings.enable_prompt_caching:
        prompt_cache_status = "disabled"
    elif _supports_cache_control(model):
        prompt_cache_status = "enabled (cache_control markers)"
    else:
        prompt_cache_status = "enabled (provider-automatic, no markers)"
    logger.info(
        f"Configured DSPy with model: {model} "
        f"(TwoStepAdapter with extraction_model={extraction_model}, "
        f"max_tokens={_resolve_max_tokens(defaults.model, defaults.max_tokens)}, "
        f"timeout={settings.llm_timeout}s, retries={settings.llm_retries}, "
        f"provider prompt caching {prompt_cache_status})"
    )


def verify_model_access(settings: Settings) -> tuple[bool, str]:
    """Verify that all configured models are accessible.

    Checks the default model, all per-signature model overrides, and the
    memory reflection models, so a typo in any of them fails fast at startup
    rather than mid-review.

    Args:
        settings: Application settings containing model configuration.

    Returns:
        Tuple of (success, message)
    """
    # Collect all unique models from config
    models_to_check: set[str] = {settings.default_model}
    
    # Check all signature-specific models
    for sig_name, sig_config in settings.signatures.items():
        if sig_config.model:
            models_to_check.add(sig_config.model)

    # Global extraction model (if different from default_model)
    if settings.extraction_model:
        models_to_check.add(settings.extraction_model)

    for module in REFLECTION_MODULES:
        reflection = settings.get_llm_config(module)
        models_to_check.add(reflection.model)


    # Check each model
    verified: list[str] = []
    failed: list[str] = []
    
    for model in models_to_check:
        try:
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            verified.append(model)
            logger.info(f"Model verified: {model}")
        except litellm.AuthenticationError as e:
            failed.append(f"{model}: authentication failed - {e}")
        except litellm.RateLimitError as e:
            failed.append(f"{model}: rate limit exceeded - {e}")
        except litellm.APIConnectionError as e:
            failed.append(f"{model}: connection error - {e}")
        except Exception as e:
            failed.append(f"{model}: {e}")
    
    if failed:
        return False, f"Model verification failed: {'; '.join(failed)}"
    
    return True, f"Verified {len(verified)} model(s): {', '.join(verified)}"


class _TaskDestroyedFilter(logging.Filter):
    """Filter to suppress 'Task was destroyed' messages from asyncio."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Task was destroyed" in msg and "LoggingWorker" in msg:
            return False
        return True


class _MCPRequestFilter(logging.Filter):
    """Filter to suppress all noisy 'Processing request of type' MCP server messages."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        return "Processing request of type" not in record.getMessage()


# Suppress LiteLLM's async logging worker warnings that occur during multi-threaded execution
logging.getLogger("asyncio").addFilter(_TaskDestroyedFilter())

# Suppress noisy MCP server "Processing request" messages
logging.getLogger("mcp.server").addFilter(_MCPRequestFilter())
logging.getLogger("mcp.server.lowlevel").addFilter(_MCPRequestFilter())
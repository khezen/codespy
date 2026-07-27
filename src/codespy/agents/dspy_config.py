"""DSPy and LiteLLM configuration utilities."""

import logging

import dspy  # type: ignore[import-untyped]
from dspy.adapters.two_step_adapter import TwoStepAdapter  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]

from codespy.config import Settings, get_settings
from codespy.config_memory import LLMSettings, REFLECTION_MODULES


logger = logging.getLogger(__name__)


def new_lm(settings: Settings, config: LLMSettings) -> dspy.LM:
    """Build a ``dspy.LM`` for resolved LLM settings.

    The single place LMs are constructed, so the cross-cutting concerns
    (timeout, retries, provider-side prompt caching) are applied uniformly.

    ``reasoning_effort`` is forwarded to LiteLLM through ``dspy.LM``'s
    ``**kwargs``; LiteLLM maps it onto each provider's native parameter
    (Anthropic thinking budget, OpenAI reasoning effort, ...).

    Args:
        settings: Application settings (timeout / retries / prompt caching).
        config: Resolved settings from ``Settings.get_llm_config()``.

    Returns:
        A configured ``dspy.LM``.
    """
    lm_kwargs: dict = {
        "model": config.model,
        "temperature": config.temperature,
        "reasoning_effort": config.reasoning_effort,
        "timeout": settings.llm_timeout,
        "num_retries": settings.llm_retries,
    }
    # Cache system prompts on the provider's servers (Anthropic, OpenAI, Bedrock...)
    if settings.enable_prompt_caching:
        lm_kwargs["cache_control_injection_points"] = [
            {"location": "message", "role": "system"}
        ]
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
    return dspy.context(lm=new_lm(settings, settings.get_llm_config(name)))



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

    # Extraction LM for TwoStepAdapter's second stage: a smaller/faster model
    # that pulls structured fields out of the main LM's free-form response.
    extraction_model = defaults.extraction_model
    extraction_lm = new_lm(
        settings, defaults.model_copy(update={"model": extraction_model})
    )


    dspy.settings.configure(
        lm=lm,
        adapter=TwoStepAdapter(extraction_lm),  # TwoStepAdapter solves ChatAdapter parsing failures
    )

    # Enable memory-only caching for LLM calls (no disk caching)
    dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=False, memory_max_entries=10000)

    prompt_cache_status = "enabled" if settings.enable_prompt_caching else "disabled"
    logger.info(
        f"Configured DSPy with model: {model} "
        f"(TwoStepAdapter with extraction_model={extraction_model}, "
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

    # Check the Hippocampus reflection models (Distiller / Cartographer)
    for module in REFLECTION_MODULES:
        reflection = settings.get_llm_config(module)
        models_to_check.add(reflection.model)
        models_to_check.add(reflection.extraction_model)


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
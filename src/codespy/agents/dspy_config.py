"""DSPy and LiteLLM configuration utilities."""

import logging

import dspy  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]

from codespy.config import Settings

logger = logging.getLogger(__name__)


def configure_dspy(settings: Settings) -> None:
    """Configure DSPy with the LLM backend.

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
    # Configure DSPy with LiteLLM
    lm = dspy.LM(model=model)
    dspy.configure(lm=lm)
    # Enable memory-only caching for LLM calls (no disk caching)
    dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=False, memory_max_entries=1000)
    logger.info(f"Configured DSPy with model: {model} (memory cache enabled)")


def verify_model_access(settings: Settings) -> tuple[bool, str]:
    """Verify that all configured models are accessible.

    Checks the default model and all per-signature model overrides.

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
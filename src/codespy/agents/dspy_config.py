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
    model = settings.litellm_model

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
    """Verify that the model is accessible.

    Args:
        settings: Application settings containing model configuration.

    Returns:
        Tuple of (success, message)
    """
    model = settings.litellm_model
    try:
        # Make a minimal test call
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        return True, "Model access verified"
    except litellm.AuthenticationError as e:
        return False, f"Authentication failed: {e}"
    except litellm.RateLimitError as e:
        return False, f"Rate limit exceeded: {e}"
    except litellm.APIConnectionError as e:
        return False, f"Connection error: {e}"
    except Exception as e:
        return False, f"Model access error: {e}"


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
"""DSPy and LiteLLM configuration utilities."""

import logging

import dspy
import litellm

from codespy.agents.cost_tracker import get_cost_tracker
from codespy.config import Settings

logger = logging.getLogger(__name__)


def _litellm_success_callback(kwargs, completion_response, start_time, end_time):
    """Callback for successful LiteLLM calls to track costs."""
    cost_tracker = get_cost_tracker()
    try:
        # Get cost from response (LiteLLM provides this)
        cost = kwargs.get("response_cost", 0.0)
        if cost == 0.0 and hasattr(completion_response, "_hidden_params"):
            cost = getattr(completion_response._hidden_params, "response_cost", 0.0) or 0.0

        # Calculate tokens
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(completion_response, "usage") and completion_response.usage:
            prompt_tokens = getattr(completion_response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(completion_response.usage, "completion_tokens", 0) or 0

        total_tokens = prompt_tokens + completion_tokens

        # Try to get cost from litellm's cost calculator if not available
        if cost == 0.0 and total_tokens > 0:
            try:
                model = kwargs.get("model", "")
                cost = litellm.completion_cost(
                    model=model,
                    prompt=str(prompt_tokens),
                    completion=str(completion_tokens),
                )
            except Exception:
                pass

        cost_tracker.add_call(cost, total_tokens)

        logger.debug(
            f"LLM call: {total_tokens} tokens, ${cost:.4f} "
            f"(total: {cost_tracker.call_count} calls, ${cost_tracker.total_cost:.4f})"
        )
    except Exception as e:
        logger.debug(f"Cost tracking error: {e}")


def get_litellm_success_callback():
    """Get the litellm success callback function for cost tracking."""
    return _litellm_success_callback


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

    # Register cost tracking callback
    if _litellm_success_callback not in litellm.success_callback:
        litellm.success_callback.append(_litellm_success_callback)

    # Configure DSPy with LiteLLM
    lm = dspy.LM(model=model)
    dspy.configure(lm=lm)

    logger.info(f"Configured DSPy with model: {model}")


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
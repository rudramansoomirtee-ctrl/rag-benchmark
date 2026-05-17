"""LiteLLM-based LLM client.

One `generate()` call works against Bedrock, Anthropic direct, OpenAI, Ollama —
swap the provider with one env var. Cost is auto-tracked by litellm.
"""
from litellm import completion

from src.config import settings


def generate(messages: list[dict], **overrides) -> dict:
    """Call the configured LLM. Returns content, token counts, and cost."""
    response = completion(
        model=settings.litellm_model,
        messages=messages,
        aws_region_name=settings.aws_region,
        temperature=0,  # deterministic for evaluation
        **overrides,
    )
    return {
        "content": response.choices[0].message.content,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
        "cost_usd": float(response._hidden_params.get("response_cost") or 0.0),
        "raw": response,
    }

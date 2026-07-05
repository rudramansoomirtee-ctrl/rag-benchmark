"""LiteLLM-based LLM client.

One `generate()` call works against Bedrock, Anthropic direct, OpenAI, Ollama —
swap the provider with one env var. Cost is auto-tracked by litellm.
"""
import instructor
from litellm import completion

from src.config import settings


def generate(messages: list[dict], **overrides) -> dict:
    """Call the configured LLM. Returns content, token counts, and cost.

    `max_tokens` defaults to `settings.max_tokens` — a hard ceiling that bounds
    the cost/latency blast radius of a generation degeneracy (observed: DeepSeek-V3
    occasionally spirals into repeating "[chunk-N][chunk-N+1]..." indefinitely
    after already stating the answer — see config.py). Callers can still override.
    """
    overrides.setdefault("max_tokens", settings.max_tokens)
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


def make_instructor_client():
    """Instructor client configured for the active LITELLM_MODEL.

    Anthropic models use TOOLS mode (the instructor default, validated for our
    schemas across exp ≤ 9). Amazon Nova on Bedrock doesn't reliably emit a
    single tool call for our multi-field schemas — it falls back to free-text
    `<answer>...</answer>` output, which instructor cannot parse. JSON mode
    bypasses tool calling entirely and asks the model for a JSON object directly
    in the response text, which Nova handles reliably.

    Note: `max_retries` is a `.create()` arg, not a constructor arg. Per-call
    overrides happen at the call sites; instructor's default of 3 retries is
    fine for our schemas under JSON mode.
    """
    model = settings.litellm_model.lower()
    if "nova" in model:
        return instructor.from_litellm(completion, mode=instructor.Mode.JSON)
    return instructor.from_litellm(completion)

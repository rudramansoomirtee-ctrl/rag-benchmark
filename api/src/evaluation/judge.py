"""LLM-as-judge answer scoring — a SECONDARY metric to substring containment.

Primary correctness stays `contains_match` (the MultiHop-RAG paper's containment
metric, Tang & Yang 2024). This adds the CRAG rubric (Yang et al. 2024) as a
second opinion: a single typed label per answer, graded by the same Bedrock model
used everywhere else. Typed via `instructor` so the label can never parse-fail;
temperature 0 to be as deterministic as the provider allows — it is non-deterministic
in principle, which is exactly why it is secondary rather than the headline number.
"""
import instructor
from litellm import completion

from src.config import settings
from src.systems.schemas import JudgeLabel, JudgeVerdict


# CRAG truthfulness weights (Yang et al. 2024): hallucination (incorrect) is
# penalised harder than abstention (missing).
CRAG_SCORE: dict[str, float] = {
    JudgeLabel.PERFECT.value: 1.0,
    JudgeLabel.ACCEPTABLE.value: 0.5,
    JudgeLabel.MISSING.value: 0.0,
    JudgeLabel.INCORRECT.value: -1.0,
}


JUDGE_SYSTEM_PROMPT = (
    "You grade a generated answer against the ground-truth answer using the CRAG "
    "rubric. Assign exactly one label:\n"
    "\n"
    "- perfect: correctly and completely answers the question with no incorrect or "
    "hallucinated content.\n"
    "- acceptable: useful and largely correct, but with minor errors or omissions "
    "that do not substantially harm usefulness.\n"
    "- missing: does not answer — says it does not know, is empty, refuses, or errors.\n"
    "- incorrect: gives wrong, contradictory, or irrelevant information (a hallucination).\n"
    "\n"
    "Grade only against the ground truth; ignore wording and style. An answer that "
    "says the information is unavailable is 'perfect' only if the ground truth also "
    "says it is unavailable/insufficient; otherwise it is 'missing'."
)


def judge(question: str, gold: str, answer: str | None) -> tuple[str, float, float]:
    """Return (label, crag_score, cost_usd) for one answer.

    Empty answers short-circuit to 'missing' without an LLM call.
    """
    if not answer or not answer.strip():
        return JudgeLabel.MISSING.value, CRAG_SCORE[JudgeLabel.MISSING.value], 0.0

    client = instructor.from_litellm(completion)
    verdict, raw = client.chat.completions.create_with_completion(
        model=settings.litellm_model,
        response_model=JudgeVerdict,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Ground-truth answer: {gold}\n\n"
                    f"Generated answer: {answer}"
                ),
            },
        ],
    )
    cost = float((getattr(raw, "_hidden_params", None) or {}).get("response_cost") or 0.0)
    label = verdict.label.value
    return label, CRAG_SCORE[label], cost

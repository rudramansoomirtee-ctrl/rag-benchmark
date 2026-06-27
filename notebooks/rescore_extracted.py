"""Post-hoc answer-span extraction + re-scoring, to make EM / token-F1 comparable
to MuSiQue / MultiHop-RAG papers (which score short extracted answers, not the
verbose cited reasoning our systems emit).

Read-only w.r.t. runs: we never touch stored answers. For each run we ask the LLM
(temp 0) to extract the final short answer span FROM the stored response text only,
then recompute contains / EM / token-F1 on raw vs extracted. Containment stays the
deterministic primary; this is an LLM-assisted normalisation for the F1/EM column.

Run inside the api container:
  docker cp notebooks/rescore_extracted.py rag-api:/tmp/rescore.py
  docker exec -e LITELLM_MODEL=bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0 \
              -e EXPS=35,9 [-e LIMIT=3] rag-api python /tmp/rescore.py
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from collections import defaultdict

from sqlalchemy import select

from src.db.models import Experiment, Query, Run
from src.db.session import get_session
from src.evaluation.metrics import contains_match, exact_match, token_f1
from src.llm.client import generate

EXPS = [int(x) for x in os.environ.get("EXPS", "35,9").split(",")]
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0 = per-system cap (smoke test) + print examples

EXTRACT_SYS = (
    "You extract the final short answer from a verbose model response to a question. "
    "Output ONLY the answer span as supported by the response text — a few words at most, "
    "no explanation, no citations, no surrounding punctuation. "
    "If the response says the answer cannot be found, or refuses, output exactly: NONE."
)


def extract(question: str, answer: str) -> tuple[str, float]:
    a = (answer or "").strip()
    if not a:
        return "", 0.0
    for attempt in range(2):
        try:
            r = generate(messages=[
                {"role": "system", "content": EXTRACT_SYS},
                {"role": "user",
                 "content": f"Question: {question}\n\nResponse:\n{a}\n\nFinal short answer:"},
            ])
            span = (r["content"] or "").strip()
            if span.lower().startswith("answer:"):
                span = span.split(":", 1)[1].strip()
            return span, float(r.get("cost_usd") or 0.0)
        except Exception as e:  # Bedrock throttle etc. — retry once, else give up
            if attempt == 0:
                continue
            print(f"  ! extract failed: {type(e).__name__}: {e}", flush=True)
            return "", 0.0
    return "", 0.0


def _pred(span: str) -> str:
    return "" if span.strip().upper() in {"NONE", "NONE.", "N/A"} else span


def score(items, key):
    n = len(items) or 1
    cm = sum(contains_match(_pred(it[key]), it["gold"] or "") for it in items) / n
    em = sum(exact_match(_pred(it[key]), it["gold"] or "") for it in items) / n
    f1 = sum(token_f1(_pred(it[key]), it["gold"] or "") for it in items) / n
    return cm, em, f1


def main():
    s = get_session()
    total_cost = 0.0
    artifact = {}
    for exp_id in EXPS:
        rows = s.execute(
            select(Run, Query, Experiment)
            .join(Query, Run.query_id == Query.id)
            .join(Experiment, Run.experiment_id == Experiment.id)
            .where(Run.experiment_id == exp_id)
            .order_by(Run.system, Run.id)
        ).all()
        if not rows:
            print(f"exp{exp_id}: no runs", flush=True)
            continue
        ds = rows[0].Query.dataset
        model = (rows[0].Experiment.config_json or {}).get("model", "?")
        bysys = defaultdict(list)
        seen = defaultdict(int)
        for r, q, _e in rows:
            if LIMIT and seen[r.system] >= LIMIT:
                continue
            seen[r.system] += 1
            span, c = extract(q.query_text, r.answer)
            total_cost += c
            rec = {"qid": q.id, "gold": q.ground_truth, "raw": r.answer or "", "extracted": span}
            bysys[r.system].append(rec)
            if LIMIT:
                print(f"  [{r.system}] gold={q.ground_truth!r}", flush=True)
                print(f"        raw[:90]={(r.answer or '')[:90]!r}", flush=True)
                print(f"        extracted={span!r}", flush=True)

        print(f"\n=== exp{exp_id}  {ds}  model={model.split('/')[-1]}  extractor={os.environ.get('LITELLM_MODEL','?').split('/')[-1]} ===", flush=True)
        print(f"{'sys':4} {'n':>3} | {'RAW contains':>12} {'EM':>6} {'F1':>6} | {'EXT contains':>12} {'EM':>6} {'F1':>6}", flush=True)
        for sysn in sorted(bysys):
            items = bysys[sysn]
            rcm, rem, rf1 = score(items, "raw")
            ecm, eem, ef1 = score(items, "extracted")
            print(f"{sysn:4} {len(items):>3} | {rcm:>12.3f} {rem:>6.3f} {rf1:>6.3f} | {ecm:>12.3f} {eem:>6.3f} {ef1:>6.3f}", flush=True)
        artifact[exp_id] = {"dataset": ds, "model": model, "runs": {k: v for k, v in bysys.items()}}

    print(f"\nTOTAL extraction cost: ${total_cost:.4f}", flush=True)
    if not LIMIT:
        os.makedirs("/data/results", exist_ok=True)
        path = f"/data/results/rescore_extracted_{'_'.join(map(str, EXPS))}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        print(f"saved spans -> {path}", flush=True)


if __name__ == "__main__":
    main()

"""Prototype: decompose-and-SOLVE F (F-solve) vs naive A, on hard multi-hop MuSiQue.

Current F decomposes RETRIEVAL then answers once — but retrieval is already
near-ceiling (Hits@10~0.95), so F barely beats A. F-solve decomposes the
REASONING: answer each sub-question over its own retrieval (sequentially, so a
later hop can use an earlier answer), then compose the final answer from the
resolved chain. This targets the real bottleneck (the budget model's multi-hop
synthesis). Reuses the SAME decomposer (_decompose) + retriever + answer prompt
as the baselines, so the only changed variable is "solve sub-Qs vs fuse-and-answer".

Cost-aware: sub-answers are short (max_tokens cap); cost summed across every call.
Scoring: alias-aware containment (+ EM/F1) on the post-'Final answer:' span — same
for A and F-solve.

Run (MuSiQue needs its own index):
  docker cp notebooks/proto_f_solve.py rag-api:/tmp/proto.py
  docker exec -e OPENSEARCH_INDEX=rag-chunks-musique -e N=24 -e HOPS=3,4 \
    -e LITELLM_MODEL=bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0 \
    rag-api python /tmp/proto.py
"""
import json
import os
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select

from src.db.models import Query
from src.db.session import get_session
from src.evaluation.metrics import _post_marker, contains_match, exact_match, token_f1
from src.llm.client import generate
from src.retrieval.retrieve import FUSED_ANSWER_TOP_K, format_context, retrieve, rrf_fuse
from src.systems.system_a import ANSWER_SYSTEM_PROMPT, SystemA
from src.systems.system_f import _decompose

N = int(os.environ.get("N", "24"))
HOPS = os.environ.get("HOPS", "3,4").split(",")
SMOKE = int(os.environ.get("SMOKE", "0"))

SUBANSWER_PROMPT = (
    "You answer ONE single-hop factual sub-question using ONLY the provided context. "
    "Reply with just the short answer (a few words) — no explanation, no citations. "
    "If the context does not contain it, reply 'UNKNOWN'."
)


def f_solve(query: str):
    """Decompose -> solve each sub-Q (sequential) -> compose. Returns (answer, cost, n_subs)."""
    cost = 0.0
    subs, _tin, _tout, dcost = _decompose(query)
    cost += dcost
    all_hits = [retrieve(query)]
    qa = []
    for sq in subs:
        hits = retrieve(sq)
        all_hits.append(hits)
        prior = "".join(f"- {q} -> {a}\n" for q, a in qa)
        user = (f"Known so far:\n{prior}\n" if prior else "") + f"Context:\n{format_context(hits)}\n\nSub-question: {sq}"
        r = generate(
            messages=[{"role": "system", "content": SUBANSWER_PROMPT},
                      {"role": "user", "content": user}],
            max_tokens=120,
        )
        cost += float(r.get("cost_usd") or 0.0)
        qa.append((sq, (r["content"] or "").strip()))
    fused = rrf_fuse(all_hits)[:FUSED_ANSWER_TOP_K]
    chain = "".join(f"- {q} -> {a}\n" for q, a in qa)
    user = (f"Context:\n{format_context(fused)}\n\n"
            + (f"Resolved sub-questions:\n{chain}\n" if chain else "")
            + f"Question: {query}")
    r = generate(messages=[{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                           {"role": "user", "content": user}])
    cost += float(r.get("cost_usd") or 0.0)
    return r["content"], cost, len(subs)


def correct(answer: str, gold: str, aliases) -> bool:
    cands = [gold] + list(aliases or [])
    return any(contains_match(answer, c) for c in cands if c)


def best_em_f1(answer: str, gold: str, aliases):
    cands = [c for c in [gold] + list(aliases or []) if c]
    post = _post_marker(answer or "")
    em = max((exact_match(post, c) for c in cands), default=False)
    f1 = max((token_f1(answer or "", c) for c in cands), default=0.0)
    return int(em), f1


def main():
    s = get_session()
    rows = s.scalars(select(Query).where(Query.dataset == "musique").order_by(Query.id)).all()
    hard = [q for q in rows if any(q.external_id.startswith(f"{h}hop") for h in HOPS)][:N]
    print(f"hard multi-hop MuSiQue subset: n={len(hard)} (hops={HOPS})\n", flush=True)

    a_sys = SystemA()
    agg = {"A": {"corr": 0, "cost": 0.0, "em": 0, "f1": 0.0},
           "F-solve": {"corr": 0, "cost": 0.0, "em": 0, "f1": 0.0, "subs": 0}}
    for i, q in enumerate(hard, 1):
        aliases = (q.query_metadata or {}).get("answer_aliases") or []
        ar = a_sys.answer(q.query_text)
        a_ok = correct(ar.answer, q.ground_truth, aliases)
        a_em, a_f1 = best_em_f1(ar.answer, q.ground_truth, aliases)
        agg["A"]["corr"] += a_ok; agg["A"]["cost"] += float(ar.cost_usd or 0.0)
        agg["A"]["em"] += a_em; agg["A"]["f1"] += a_f1

        fa, fcost, nsub = f_solve(q.query_text)
        f_ok = correct(fa, q.ground_truth, aliases)
        f_em, f_f1 = best_em_f1(fa, q.ground_truth, aliases)
        agg["F-solve"]["corr"] += f_ok; agg["F-solve"]["cost"] += fcost
        agg["F-solve"]["em"] += f_em; agg["F-solve"]["f1"] += f_f1; agg["F-solve"]["subs"] += nsub

        flag = {(True, False): " <-- F-solve wins", (False, True): " <-- A wins"}.get((f_ok, a_ok), "")
        print(f"[{i}/{len(hard)}] {q.external_id[:14]} gold={q.ground_truth[:34]!r}  A={'Y' if a_ok else 'n'} F-solve={'Y' if f_ok else 'n'}{flag}", flush=True)
        if SMOKE:
            print(f"      A.ans={_post_marker(ar.answer).strip()[:70]!r}", flush=True)
            print(f"      F.subs={nsub}  F.ans={_post_marker(fa).strip()[:70]!r}", flush=True)

    n = len(hard) or 1
    print(f"\n{'system':9} {'acc':>6} {'EM':>6} {'F1':>6} {'$/query':>9} {'$/correct':>10}", flush=True)
    for name in ("A", "F-solve"):
        d = agg[name]
        acc = d["corr"] / n
        cpc = (d["cost"] / d["corr"]) if d["corr"] else float("inf")
        extra = f"  (avg {d['subs']/n:.1f} sub-Qs)" if name == "F-solve" else ""
        print(f"{name:9} {acc:>6.3f} {d['em']/n:>6.3f} {d['f1']/n:>6.3f} {d['cost']/n:>9.5f} {cpc:>10.5f}{extra}", flush=True)

    if not SMOKE:
        os.makedirs("/data/results", exist_ok=True)
        with open("/data/results/proto_f_solve.json", "w") as f:
            json.dump({"n": n, "hops": HOPS, "agg": agg}, f, indent=2)
        print("\nsaved -> /data/results/proto_f_solve.json", flush=True)


if __name__ == "__main__":
    main()

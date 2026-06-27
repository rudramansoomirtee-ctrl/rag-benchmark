"""Measure the two legitimate accuracy levers in ONE process (reranker loads once):

  Change 1 — terminal 'Final answer:' prompt: confirm System A now emits the
             marker and that contains/EM/token-F1 score the committed span.
  Change 2 — retrieval pool size: recall@10 / MRR@10 / Hits@10 at pool=20 vs the
             values in POOLS env, retrieval-only (no LLM), on multihop.

Run in the api container (multihop uses the default index, no switching):
  docker cp notebooks/tune_levers.py rag-api:/tmp/tune.py
  docker exec -e N=15 -e POOLS=20,50 \
    -e LITELLM_MODEL=bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0 \
    rag-api python /tmp/tune.py
"""
import os
import sys
import time

sys.path.insert(0, "/app")
from sqlalchemy import select

from src.config import settings
from src.db.models import Query
from src.db.session import get_session
from src.evaluation.metrics import (
    _post_marker, contains_match, exact_match, hit_at_k, recall_at_k,
    reciprocal_rank_at_k, token_f1,
)
from src.retrieval.retrieve import retrieve
from src.systems.system_a import ANSWER_SYSTEM_PROMPT, SystemA

N = int(os.environ.get("N", "15"))
POOLS = [int(x) for x in os.environ.get("POOLS", "20,50").split(",")]
CONFIRM = int(os.environ.get("CONFIRM", "5"))

s = get_session()
qs = [q for q in s.scalars(
    select(Query).where(Query.dataset == "multihop", Query.split == "eval").order_by(Query.id)
).all() if q.relevant_chunk_ids][:N]
print(f"loaded {len(qs)} scorable multihop queries\n", flush=True)

print("=== Change 2: retrieval pool lever (multihop, retrieval-only) ===", flush=True)
for pool in POOLS:
    settings.retrieval_pool = pool
    mrr = r5 = r10 = h4 = h10 = 0.0
    t0 = time.time()
    for q in qs:
        ranked = [h["chunk_id"] for h in retrieve(q.query_text, top_k=10)]
        mrr += reciprocal_rank_at_k(ranked, q.relevant_chunk_ids, 10)
        r5 += recall_at_k(ranked, q.relevant_chunk_ids, 5)
        r10 += recall_at_k(ranked, q.relevant_chunk_ids, 10)
        h4 += hit_at_k(ranked, q.relevant_chunk_ids, 4)
        h10 += hit_at_k(ranked, q.relevant_chunk_ids, 10)
    n = len(qs)
    print(f"  pool={pool:>3}: MRR@10={mrr/n:.3f}  R@5={r5/n:.3f}  R@10={r10/n:.3f}  "
          f"Hits@4={h4/n:.3f}  Hits@10={h10/n:.3f}   ({(time.time()-t0)/n:.1f}s/q)", flush=True)
settings.retrieval_pool = POOLS[0]

print(f"\n=== Change 1: terminal 'Final answer:' prompt (System A, {CONFIRM} queries) ===", flush=True)
print(f"prompt tail: ...{ANSWER_SYSTEM_PROMPT[-90:]!r}\n", flush=True)
a = SystemA()
emits = 0
for q in qs[:CONFIRM]:
    r = a.answer(q.query_text)
    full = r.answer or ""
    post = _post_marker(full)
    has_marker = "final answer:" in full.lower()
    emits += has_marker
    print(f"gold={q.ground_truth!r}  marker={has_marker}", flush=True)
    print(f"  committed={post.strip()[:90]!r}", flush=True)
    print(f"  contains={contains_match(full, q.ground_truth)} "
          f"EM={exact_match(post, q.ground_truth)} F1={token_f1(full, q.ground_truth):.2f}", flush=True)
print(f"\nmarker emitted in {emits}/{CONFIRM} answers", flush=True)

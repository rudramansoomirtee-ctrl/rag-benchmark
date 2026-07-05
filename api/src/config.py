"""Application config — single source of truth, loaded from environment."""
import litellm
from pydantic_settings import BaseSettings, SettingsConfigDict

# Drop provider-unsupported params silently (e.g. Anthropic-style `tool_choice`
# isn't accepted by Amazon Nova). With this off, swapping LITELLM_MODEL to a
# non-Anthropic Bedrock model raises mid-experiment in any system that uses
# `instructor` (B/F/G/judge). With it on, Haiku is unaffected (all params remain
# valid) and Nova/DeepSeek/etc. just work. Set once at module import.
litellm.drop_params = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infra
    database_url: str = "postgresql+psycopg://rag:ragbench@postgres:5432/ragbench"
    opensearch_url: str = "http://opensearch:9200"
    phoenix_collector_endpoint: str = "http://phoenix:6006"

    # LLM
    litellm_model: str = "bedrock/amazon.nova-lite-v1:0"
    # LLM-as-judge model — independent of generation so you can run cheap
    # generation + strong judging. Falls back to litellm_model when unset.
    judge_model: str | None = None
    aws_region: str = "eu-west-2"

    # Retrieval
    embedding_model: str = "BAAI/llm-embedder"
    embedding_dim: int = 768
    opensearch_index: str = "rag-chunks"
    # Answer-context budget = 20 for EVERY system: A/A-minus answer over their
    # single-pass top-20, B/F/F-seq over their fused top-20 (= fused_answer_top_k).
    # Raised 10→20 so the budget is UNIFORM across the whole factorial (removes the
    # earlier A=10 vs fusing=20 asymmetry), and because chunk-level analysis of
    # exp36/37 showed gold being evicted from a 10-slot fused context (the dominant
    # 4-hop failure). Env: TOP_K.
    top_k: int = 20
    # Answer-context budget for systems fusing multiple ranked lists (B, F, F-seq):
    # they answer over the fused top-N, held CONSTANT at 20 so the comparison isolates
    # orchestration strategy, not the budget knob — and now equal to A's top_k=20, so
    # the budget is uniform across all eight systems. NB the budget ablation
    # (exp38/39/40) found the optimum is per-strategy (B best @10, F-seq @20); a single
    # fixed value is adopted for a controlled comparison. Env: FUSED_ANSWER_TOP_K.
    fused_answer_top_k: int = 20
    # First-stage hybrid pool handed to the cross-encoder reranker. Kept at ~2× top_k
    # (40 for top_k=20) so the reranker actually SELECTS the answer context from a
    # larger candidate set rather than just reordering exactly what it was given.
    retrieval_pool: int = 40
    # BGE-reranker-v2-m3 (568M params) — top of MTEB rerank as of 2024, free,
    # CPU-tolerable (~50-100ms per pair). Strong open-weight replacement for
    # Cohere Rerank 3.5 which is not available in eu-west-2. The smaller
    # cross-encoder/ms-marco-MiniLM-L-6-v2 (22M) was the previous default
    # (used by experiments 1-9); the swap is meaningful enough to warrant a
    # full re-run rather than a comparison across configurations.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Reranker provider: "bedrock-cohere" (Cohere Rerank 3.5 via Bedrock — the DEFAULT
    # and the reranker used for the final matrix and the pilots exp36-43) or "local"
    # (CPU cross-encoder, free, used for exp ≤9). Bedrock rerank needs the model
    # enabled in the Bedrock model-access page; failures fall back to the local
    # cross-encoder so a bad config doesn't break a long run — NB this fallback can
    # make retrieval inconsistent mid-matrix, so watch logs for it during final runs.
    # Cohere rerank is a METERED Bedrock API call NOT captured in runs.cost_usd.
    rerank_provider: str = "bedrock-cohere"
    bedrock_rerank_model_id: str = "cohere.rerank-v3-5:0"
    # Bedrock Rerank isn't offered in eu-west-2; call it in a supported region
    # (eu-central-1 Frankfurt) while LLM generation stays in aws_region.
    bedrock_rerank_region: str = "eu-central-1"
    # Source-stratified first-stage pool (shared-retriever ablation lever).
    # When on, retrieve() guarantees each source's top candidate enters the
    # rerank pool so one publisher can't monopolise it — the generic,
    # source-agnostic replacement for F-tuned's reserved-slot hack. Applied in
    # the shared retrieve() so A/B/F/F-tuned inherit it identically, keeping the
    # controlled comparison valid. OFF = legacy behaviour (env: RETRIEVAL_STRATIFY_SOURCES).
    retrieval_stratify_sources: bool = False
    # Naive dense-kNN-only retriever (no BM25 / RRF / rerank) for weakened-retriever
    # ablations. OFF = full hybrid+rerank pipeline (env: RETRIEVAL_SEMANTIC_ONLY).
    retrieval_semantic_only: bool = False

    # Agent
    max_agent_steps: int = 5


settings = Settings()

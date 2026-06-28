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
    top_k: int = 10
    # Answer-context budget for systems fusing multiple ranked lists (B, F, F-seq):
    # they answer over the fused top-N. Raised 10→20 after chunk-level analysis of
    # exp36/37 showed gold chunks that WERE retrieved getting evicted from a 10-slot
    # fused context as iterations/sub-questions piled up — the dominant 4-hop failure
    # mode ("retrieved but not in answer context"). A single-list (A) run answers
    # over top_k directly and is unaffected. Env: FUSED_ANSWER_TOP_K.
    # Held CONSTANT across all fusing systems (B, F, F-seq) so the comparison
    # isolates orchestration strategy, not the budget knob. NB the ablation
    # (exp38/39/40) showed the optimum is per-strategy — B@10=0.600 > B@20=0.540,
    # but F-seq@20=0.540 >> F-seq@10=0.380 — so a single fixed budget trades ~0.06
    # of B's accuracy for a clean, budget-controlled comparison.
    fused_answer_top_k: int = 20
    retrieval_pool: int = 20
    # BGE-reranker-v2-m3 (568M params) — top of MTEB rerank as of 2024, free,
    # CPU-tolerable (~50-100ms per pair). Strong open-weight replacement for
    # Cohere Rerank 3.5 which is not available in eu-west-2. The smaller
    # cross-encoder/ms-marco-MiniLM-L-6-v2 (22M) was the previous default
    # (used by experiments 1-9); the swap is meaningful enough to warrant a
    # full re-run rather than a comparison across configurations.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Reranker provider: "local" (CPU cross-encoder, free, used for exp ≤9) or
    # "bedrock-cohere" (Cohere Rerank 3.5 via Bedrock — set RERANK_PROVIDER=bedrock-cohere
    # in .env to switch). Bedrock rerank needs the model enabled in the Bedrock
    # model-access page; failures fall back to the local cross-encoder so a bad
    # config doesn't break a long run.
    rerank_provider: str = "local"
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

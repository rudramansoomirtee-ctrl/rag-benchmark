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
    top_k: int = 5
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

    # Agent
    max_agent_steps: int = 5

    # HHEM faithfulness gate threshold.
    # Empirically HHEM-2.1-open scores news-domain (MultiHop) answers in [0.05, 0.30];
    # the natural 0.5 default would flag effectively every answer. 0.10 retains
    # meaningful discrimination — only the most clearly ungrounded answers flag.
    hhem_threshold: float = 0.10


settings = Settings()

"""Application config — single source of truth, loaded from environment."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infra
    database_url: str = "postgresql+psycopg://rag:ragbench@postgres:5432/ragbench"
    opensearch_url: str = "http://opensearch:9200"
    phoenix_collector_endpoint: str = "http://phoenix:6006"

    # LLM
    litellm_model: str = "bedrock/amazon.nova-lite-v1:0"
    aws_region: str = "eu-west-2"

    # Retrieval
    embedding_model: str = "BAAI/llm-embedder"
    embedding_dim: int = 768
    opensearch_index: str = "rag-chunks"
    top_k: int = 5
    retrieval_pool: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Agent
    max_agent_steps: int = 5

    # HHEM faithfulness gate threshold.
    # Empirically HHEM-2.1-open scores news-domain (MultiHop) answers in [0.05, 0.30];
    # the natural 0.5 default would flag effectively every answer. 0.10 retains
    # meaningful discrimination — only the most clearly ungrounded answers flag.
    hhem_threshold: float = 0.10


settings = Settings()

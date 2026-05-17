"""Application config — single source of truth, loaded from environment."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infra
    database_url: str = "postgresql+psycopg://rag:ragbench@postgres:5432/ragbench"
    opensearch_url: str = "http://opensearch:9200"
    phoenix_collector_endpoint: str = "http://phoenix:6006"

    # LLM
    litellm_model: str = "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
    aws_region: str = "eu-west-2"

    # Retrieval
    embedding_model: str = "BAAI/llm-embedder"
    embedding_dim: int = 768  # llm-embedder is 768-dim
    opensearch_index: str = "rag-chunks"
    top_k: int = 5

    # Agent
    max_agent_steps: int = 5

    # HHEM faithfulness gate threshold — populated by calibrate command
    hhem_threshold: float = 0.5


settings = Settings()

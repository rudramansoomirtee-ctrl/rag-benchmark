"""Embedding model wrapper. Loaded once, cached for the process lifetime."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model. Cached after first call."""
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns dense vectors."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]

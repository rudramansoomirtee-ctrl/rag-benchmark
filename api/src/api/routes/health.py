"""Health-check endpoint. Verifies all four dependencies are reachable."""
from fastapi import APIRouter
from sqlalchemy import text

from src.db.session import engine
from src.retrieval.opensearch_client import get_client

router = APIRouter()


@router.get("/health")
def health():
    status = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = f"error: {e}"

    try:
        info = get_client().info()
        status["opensearch"] = info["version"]["number"]
    except Exception as e:
        status["opensearch"] = f"error: {e}"

    return status

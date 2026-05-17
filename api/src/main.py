"""FastAPI entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.tracing import init_tracing
from src.api.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    yield


app = FastAPI(
    title="RAG Benchmark API",
    description="Evaluator API for Systems A/B/C/D against MultiHop-RAG and RAGTruth.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)

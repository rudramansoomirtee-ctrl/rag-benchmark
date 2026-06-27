"""FastAPI entrypoint."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.routes import ask, data, experiments, health
from src.tracing import init_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    yield


app = FastAPI(
    title="RAG Benchmark API",
    description="Evaluator API for Systems A/B/F against MultiHop-RAG and MuSiQue.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(data.router)
app.include_router(experiments.router)


_INDEX_HTML = Path(__file__).parent / "api" / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return _INDEX_HTML.read_text()

FROM python:3.11-slim

# uv from the official image — single binary, 10-30x faster than pip
COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download HHEM weights AND tokenizer so first runtime call is instant.
# Both are needed; downloading only the model leaves the tokenizer fetch for run time.
ENV HF_HOME=/models
RUN python -c "from transformers import AutoModelForSequenceClassification; \
    AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)"

# Pre-download the embedding model too — same logic
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/llm-embedder')"

COPY src/ ./src/
# Vendored OpenRag engine for System E — top-level packages so the package's
# own absolute imports (`from knowledge_base...`, `from ultimate_rag...`) resolve.
COPY ultimate_rag/ ./ultimate_rag/
COPY knowledge_base/ ./knowledge_base/
COPY alembic.ini .
COPY pyproject.toml .

EXPOSE 8000

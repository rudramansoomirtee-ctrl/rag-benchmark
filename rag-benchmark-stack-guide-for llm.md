# RAG Benchmark Stack — Technical Guide v2

**Purpose:** Reproducible, Dockerised, full-stack environment to run Systems A/B/C/D against MultiHop-RAG and RAGTruth, capture every metric to a database, trace every LLM/agent call, and surface results in a live dashboard.

**Design principles:** mirror your AWS prod patterns locally (OpenSearch, FastAPI, LangGraph), lean on best-in-class OSS instead of bespoke code, keep services minimal, zero cloud spend during eval, examiner can `docker compose up` and reproduce.

**What changed from v1:** Streamlit dashboard removed. Phoenix (Arize) added for OpenTelemetry-based LLM tracing + eval dashboard. LiteLLM SDK replaces the custom Bedrock client and the custom cost tracker. `uv` replaces `pip` in the Dockerfile. Instructor replaces ad-hoc JSON parsing for agent decisions. Rich added to the CLI. Marimo recommended for the post-eval analysis notebook. Net effect: ~150 fewer lines of code you have to write and defend.

---

## 1. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host (laptop / EC2 t3.medium for the full run)                     │
│                                                                      │
│  ┌─────────────────┐   ┌──────────────────────────────────────┐    │
│  │ phoenix         │◀──│ api                                  │    │
│  │ traces + evals  │   │ FastAPI + LangGraph + HHEM           │    │
│  │ + dashboard UI  │   │ + evaluator CLI                      │    │
│  │ :6006           │   │ uses litellm SDK to call Bedrock     │    │
│  └─────────────────┘   │ :8000                                │    │
│                        └──┬──────────────────┬────────────────┘    │
│                           │                  │                      │
│                           ▼                  ▼                      │
│                  ┌──────────────────┐  ┌──────────────────┐        │
│                  │ opensearch       │  │ postgres         │        │
│                  │ retrieval        │  │ structured runs  │        │
│                  │ (mirrors AWS)    │  │ + metrics        │        │
│                  │ :9200            │  │ :5432            │        │
│                  └──────────────────┘  └──────────────────┘        │
│                           │                                         │
│                           ▼ HTTPS                                   │
│                  AWS Bedrock (Claude Haiku 4.5)                     │
│                  cost auto-tracked by litellm, ~$3-5 total          │
└─────────────────────────────────────────────────────────────────────┘
```

**4 services. No frontend container** — Phoenix replaces what Streamlit was doing.

Why each is here:
- **opensearch** — same software as your AWS prod, identical query API, free locally
- **postgres** — durable storage, deterministic SQL aggregation for dissertation tables
- **phoenix** — auto-instruments LangChain/LangGraph, captures every span (retrieve → reformulate → answer), logs precision@k and faithfulness against the same traces, gives you embedding visualisations and dataset versioning for free. Self-hosted, MIT-licensed.
- **api** — single Python service holding FastAPI + LangGraph + HHEM + the evaluator CLI. Uses `litellm` to call Bedrock (or anything else, one config line).

Bedrock stays external. No reason to mock it; it's already your prod LLM provider, and `litellm` tracks its cost automatically.

---

## 2. Why these specific library choices

This is the section I'd defend in a Chapter 3 viva.

| Concern | Library | Why this one |
|---|---|---|
| Retrieval engine | OpenSearch (community Docker) | Mirrors AWS OpenSearch Service exactly. Hybrid BM25 + k-NN. Identical client code in lab and prod. |
| Embeddings | `BAAI/llm-embedder` via sentence-transformers | Matches the MultiHop-RAG paper's baseline. Free, runs on CPU. Comparability of headline numbers. |
| LLM gateway | `litellm` SDK | One `completion()` call works against Bedrock, Anthropic direct, OpenAI, Ollama. Cost-per-call returned in `response._hidden_params["response_cost"]`. Removes ~80 lines of custom Bedrock + cost code. |
| Agent framework | LangGraph | Already your prod framework. State machine is the right abstraction for the RETRIEVE/REFORMULATE/ANSWER loop. |
| Agent decision parsing | `instructor` (Pydantic-typed outputs) | Forces the agent's next-action choice into a typed enum. Eliminates JSON-parsing bugs that bite at run #1413. |
| Faithfulness scorer | `vectara/hallucination_evaluation_model` (HHEM-2.1-open) | Only credible open-source option. CPU-runnable. Returns calibrated 0..1 score. |
| Tracing + dashboard | Arize Phoenix | OpenTelemetry-based. Auto-instruments LangChain. Drill into any single query's span tree. Native `log_evaluations()` for precision@k. Free, self-hosted. |
| Structured storage | PostgreSQL 16 | SQL aggregation gives you any dissertation table cell in one query. Source of truth even when Phoenix's data is for debugging. |
| Dataset loading | HuggingFace `datasets` | One-line load for MultiHop-RAG. Cached on disk. |
| IR metrics | hand-rolled (~10 LOC) | precision@k and recall@k against ground-truth chunk IDs are arithmetic. No LLM-judge needed (skip Ragas here). |
| Calibration curve | `scikit-learn.metrics.roc_curve` + matplotlib | Standard, examiner-recognised, deterministic. |
| Docker dependency install | `uv` | 10-30× faster than pip. Builds your image in seconds, not minutes. |
| CLI scaffold | `typer` + `rich` | Typed commands, beautiful progress bars on the 1600-call eval loop, professional demo polish. |
| Final analysis notebook | `marimo` | Reactive, no hidden state, git-friendly. Examiner can re-run your analysis deterministically. |

What I considered and **rejected** (worth knowing why):
- **Ragas / DeepEval / TruLens for retrieval metrics** — they use LLM-as-judge, which means non-determinism and added API cost where simple arithmetic suffices. Keep ground-truth-based scoring; reserve Ragas for one ablation if you want a second opinion on answer correctness.
- **LangSmith** — same value as Phoenix but it's hosted SaaS. Phoenix self-hosted keeps your repo zero-credential and reproducible.
- **A separate vector DB (Qdrant, Weaviate, LanceDB)** — you already have OpenSearch. Two retrieval engines is one ablation; three is noise.
- **A LiteLLM proxy container** — useful for multi-user platforms. Single-user single-project: SDK is simpler. Use the proxy only if you grow beyond this.
- **Streamlit / Gradio / FastHTML** — Phoenix's UI handles everything you'd build there.
- **Prefect / Airflow / Dagster** — your evaluator is a `for` loop.

---

## 3. Directory layout

```
rag-benchmark/
├── docker-compose.yml
├── .env.example
├── .env                          # gitignored
├── README.md
│
├── api/
│   ├── Dockerfile                # uv-based, fast builds
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── alembic.ini
│   └── src/
│       ├── main.py               # FastAPI app + Phoenix register()
│       ├── config.py             # pydantic-settings
│       ├── tracing.py            # Phoenix setup, one-time init
│       ├── db/
│       │   ├── models.py         # SQLAlchemy
│       │   ├── session.py
│       │   └── migrations/       # alembic versions
│       ├── datasets/
│       │   ├── multihop.py       # load yixuantt/MultiHopRAG
│       │   └── ragtruth.py       # load ParticleMedia/RAGTruth
│       ├── retrieval/
│       │   ├── opensearch_client.py
│       │   ├── embeddings.py     # BAAI/llm-embedder
│       │   └── indexer.py        # one-shot corpus ingestion
│       ├── llm/
│       │   └── client.py         # ~15 LOC wrapping litellm.completion
│       ├── faithfulness/
│       │   └── hhem.py           # model loaded once at startup
│       ├── systems/
│       │   ├── base.py           # System protocol + RunResult
│       │   ├── system_a.py       # naive RAG (~40 LOC)
│       │   ├── system_b.py       # LangGraph agentic (~120 LOC)
│       │   ├── system_c.py       # A + HHEM gate (~10 LOC over A)
│       │   ├── system_d.py       # B + HHEM gate (~10 LOC over B)
│       │   └── schemas.py        # Instructor Pydantic models
│       ├── evaluation/
│       │   ├── metrics.py        # precision@k, recall@k
│       │   ├── calibration.py    # threshold fitting + ROC
│       │   └── runner.py         # the experiment loop, with Rich progress
│       ├── api/
│       │   └── routes/
│       └── cli.py                # `python -m src.cli ...`
│
├── opensearch/
│   └── opensearch.yml
│
├── notebooks/
│   └── analysis.py               # Marimo notebook for Chapter 4 figures
│
└── data/                         # gitignored
    ├── corpora/
    └── results/                  # exported JSONs, calibration_curve.png
```

---

## 4. Database schema

Five tables. Each `run` row now has a `phoenix_trace_id` pointer so you can jump from a SQL aggregate straight to the trace span tree.

```sql
CREATE TABLE experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    config_json     JSONB NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    notes           TEXT
);

CREATE TABLE queries (
    id                  SERIAL PRIMARY KEY,
    dataset             TEXT NOT NULL,           -- 'multihop' | 'ragtruth'
    external_id         TEXT NOT NULL,
    split               TEXT NOT NULL,           -- 'calibration' | 'eval'
    task_type           TEXT,                    -- 'qa' | 'summary' | 'data2txt'
    query_text          TEXT NOT NULL,
    ground_truth        TEXT,
    relevant_chunk_ids  JSONB NOT NULL,
    metadata            JSONB,
    UNIQUE(dataset, external_id)
);
CREATE INDEX ON queries(dataset, split);

CREATE TABLE chunks (
    id              SERIAL PRIMARY KEY,
    dataset         TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    text            TEXT NOT NULL,
    metadata        JSONB,
    UNIQUE(dataset, external_id)
);

CREATE TABLE runs (
    id                      SERIAL PRIMARY KEY,
    experiment_id           INT REFERENCES experiments(id) ON DELETE CASCADE,
    system                  TEXT NOT NULL,
    query_id                INT REFERENCES queries(id),
    retrieved_chunk_ids     JSONB NOT NULL,
    answer                  TEXT,
    hhem_score              FLOAT,
    flagged                 BOOLEAN,
    n_steps                 INT,                 -- LangGraph step count
    tokens_in               INT,
    tokens_out              INT,
    latency_ms              INT,
    cost_usd                NUMERIC(10, 6),      -- populated by litellm
    is_correct              BOOLEAN,
    phoenix_trace_id        TEXT,                -- link to span tree
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(experiment_id, system, query_id)
);
CREATE INDEX ON runs(experiment_id, system);

CREATE TABLE metrics (
    id                      SERIAL PRIMARY KEY,
    experiment_id           INT REFERENCES experiments(id) ON DELETE CASCADE,
    system                  TEXT NOT NULL,
    dataset                 TEXT NOT NULL,
    n_queries               INT NOT NULL,
    precision_at_5          FLOAT,
    recall_at_5             FLOAT,
    avg_faithfulness        FLOAT,
    pct_flagged             FLOAT,
    avg_trajectory_length   FLOAT,
    accuracy                FLOAT,
    total_cost_usd          NUMERIC(10, 4),
    cost_per_correct        NUMERIC(10, 6),
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(experiment_id, system, dataset)
);
```

**Why no `trajectory` JSONB column anymore:** the full per-step trace lives in Phoenix, indexed by `phoenix_trace_id`. Postgres keeps `n_steps` for SQL aggregation ("show me the mean trajectory length for System B on multi-hop queries"). You get the best of both: quick aggregation + rich drill-down.

---

## 5. docker-compose.yml

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:2.18.0
    container_name: rag-opensearch
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
      - DISABLE_INSTALL_DEMO_CONFIG=true
      - "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - opensearch_data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      retries: 12

  postgres:
    image: postgres:16-alpine
    container_name: rag-postgres
    environment:
      POSTGRES_DB: ragbench
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d ragbench"]
      interval: 5s
      retries: 12

  phoenix:
    image: arizephoenix/phoenix:latest
    container_name: rag-phoenix
    environment:
      PHOENIX_SQL_DATABASE_URL: postgresql://rag:${POSTGRES_PASSWORD}@postgres:5432/phoenix
      PHOENIX_WORKING_DIR: /phoenix
    ports:
      - "6006:6006"        # UI
      - "4317:4317"        # OTLP gRPC
    volumes:
      - phoenix_data:/phoenix
    depends_on:
      postgres: { condition: service_healthy }

  api:
    build: ./api
    container_name: rag-api
    depends_on:
      opensearch: { condition: service_healthy }
      postgres:   { condition: service_healthy }
      phoenix:    { condition: service_started }
    environment:
      DATABASE_URL: postgresql+psycopg://rag:${POSTGRES_PASSWORD}@postgres:5432/ragbench
      OPENSEARCH_URL: http://opensearch:9200
      PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      AWS_REGION: eu-west-2
      LITELLM_MODEL: bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
      HF_HOME: /models
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - hf_models:/models
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000

volumes:
  opensearch_data:
  pg_data:
  phoenix_data:
  hf_models:
```

Memory budget: ~3 GB OpenSearch + ~500 MB Postgres + ~500 MB Phoenix + ~2 GB api. Fits a t3.medium or any modern laptop.

**Note on Phoenix's DB:** Phoenix wants its own Postgres database (`phoenix`, not `ragbench`). Either create it in an init script, or use Phoenix's default SQLite by dropping the `PHOENIX_SQL_DATABASE_URL` env var. For this scale, SQLite is genuinely fine — switch if you want.

---

## 6. api/Dockerfile (uv-based, fast)

```dockerfile
FROM python:3.11-slim

# uv from the official image — single binary, 10-30× faster than pip
COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY src/ ./src/
COPY alembic.ini .

# Pre-download HHEM weights into the image so first run is instant
RUN python -c "from transformers import AutoModelForSequenceClassification; \
    AutoModelForSequenceClassification.from_pretrained( \
        'vectara/hallucination_evaluation_model', trust_remote_code=True)"

EXPOSE 8000
```

`requirements.txt`:

```
# Web layer
fastapi==0.115.*
uvicorn[standard]==0.32.*
pydantic-settings==2.*

# Database
sqlalchemy==2.*
psycopg[binary]==3.*
alembic==1.13.*

# Retrieval
opensearch-py==2.*
sentence-transformers==3.*

# LLM gateway — handles Bedrock + cost tracking
litellm==1.52.*

# Structured outputs for the agent
instructor==1.6.*

# Faithfulness
transformers==4.46.*
torch==2.4.*                    # CPU wheels are sufficient

# Agentic loop
langchain==0.3.*
langgraph==0.2.*

# Dataset loading
datasets==3.*

# Tracing — Phoenix + auto-instrumentation
arize-phoenix-otel==0.6.*
openinference-instrumentation-langchain==0.1.*
openinference-instrumentation-litellm==0.1.*

# Evaluation utilities
scikit-learn==1.5.*
matplotlib==3.9.*

# CLI
typer==0.13.*
rich==13.*
```

---

## 7. Tracing setup — one file, instruments everything

`api/src/tracing.py`:

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

def init_tracing(project_name: str = "rag-benchmark"):
    tracer_provider = register(
        project_name=project_name,
        endpoint="http://phoenix:6006/v1/traces",
        auto_instrument=False,                  # we'll be explicit
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider
```

Called once in `main.py` startup and once at the top of every CLI command. From that point on, every LangChain node, every `litellm.completion()` call, every chain step gets a span automatically. You don't write tracing code anywhere else.

---

## 8. The LLM client — ~15 lines, full provider portability

`api/src/llm/client.py`:

```python
from litellm import completion
from src.config import settings

def generate(messages: list[dict], **overrides) -> dict:
    response = completion(
        model=settings.litellm_model,         # bedrock/anthropic.claude-haiku-4-5-...
        messages=messages,
        aws_region_name=settings.aws_region,
        **overrides,
    )
    return {
        "content": response.choices[0].message.content,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
        "cost_usd": response._hidden_params.get("response_cost", 0.0),
        "trace_id": response._hidden_params.get("trace_id"),
    }
```

That's the whole client. To run an ablation against Sonnet, change one env var. To run against Ollama for examiners without AWS, change one env var.

---

## 9. The agent decision — typed, not string-parsed

`api/src/systems/schemas.py`:

```python
from enum import Enum
from pydantic import BaseModel, Field

class AgentAction(str, Enum):
    REFORMULATE = "reformulate"
    ANSWER = "answer"

class AgentDecision(BaseModel):
    """The agent's choice for the next step."""
    action: AgentAction = Field(description="What to do next given retrieved context.")
    reformulated_query: str | None = Field(
        default=None,
        description="Required if action=reformulate. The improved query.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Required if action=answer. The final response.",
    )
    reasoning: str = Field(description="Brief justification, ≤30 words.")
```

In System B's LangGraph node, getting a typed decision is one line:

```python
import instructor
from litellm import completion

client = instructor.from_litellm(completion)
decision = client.chat.completions.create(
    model=settings.litellm_model,
    response_model=AgentDecision,
    messages=[...],
)
# decision.action is now a typed enum, never a malformed string
```

Three benefits: (a) you can't get JSON-parse errors at query #1413, (b) Chapter 4's sequence diagram has clean labels straight from the enum, (c) Phoenix captures the typed decision as structured span attributes you can filter on.

---

## 10. The four systems share one interface

```python
# api/src/systems/base.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class RunResult:
    answer: str
    retrieved_chunk_ids: list[str]
    hhem_score: float | None
    flagged: bool | None
    n_steps: int
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    phoenix_trace_id: str | None

class System(Protocol):
    name: str
    def answer(self, query: str) -> RunResult: ...
```

Implementations: A ≈ 40 LOC, B ≈ 120 LOC, C ≈ A + 10 LOC, D ≈ B + 10 LOC. Total system code under 250 LOC. Each call is automatically traced end-to-end by Phoenix without a single explicit tracing line.

---

## 11. End-to-end workflow

```bash
# 0. Bootstrap
cp .env.example .env
docker compose up -d opensearch postgres phoenix
docker compose run --rm api alembic upgrade head

# 1. Ingest
docker compose run --rm api python -m src.cli ingest-dataset multihop
docker compose run --rm api python -m src.cli ingest-dataset ragtruth
docker compose run --rm api python -m src.cli index-corpus multihop

# 2. Calibrate HHEM on the leakage-safe RAGTruth test split
docker compose run --rm api python -m src.cli calibrate \
    --calibration-split ragtruth.calibration \
    --output /data/results/threshold.json

# 3. Run all four systems on both datasets
docker compose up -d api
docker compose exec api python -m src.cli run-experiment \
    --name "dissertation-final" \
    --systems A,B,C,D \
    --datasets multihop,ragtruth \
    --split eval
# → Rich progress bar in terminal, live spans flowing into Phoenix UI

# 4. Compute aggregates
docker compose exec api python -m src.cli compute-metrics \
    --experiment dissertation-final

# 5. Export for the dissertation
docker compose exec api python -m src.cli export \
    --experiment dissertation-final \
    --format json \
    --output /data/results/
```

During and after Step 3 you have **two views open**:
- **Phoenix at `http://localhost:6006`** — every span, every reformulation, every HHEM score, embedding scatter plots, drill-down per query. This is where you debug "why did System B fail on query 47."
- **FastAPI docs at `http://localhost:8000/docs`** — programmatic access for any custom queries.

For Chapter 4's final figures: open `notebooks/analysis.py` in Marimo (`marimo edit notebooks/analysis.py`), it reads directly from Postgres and produces the calibration curve, the Pareto plot, the trajectory-length-vs-correctness scatter — all reactive, all reproducible.

---

## 12. Configuration & secrets

`.env.example`:

```env
POSTGRES_PASSWORD=changeme
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-west-2
LITELLM_MODEL=bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
```

For an examiner-friendly variant without AWS access, ship `docker-compose.ollama.yml` that overrides `LITELLM_MODEL=ollama/llama3.1:8b` and adds an Ollama container. Same evaluator code path; numbers will differ but the pipeline reproduces.

---

## 13. Cost summary

| Item | Cost |
|------|------|
| OpenSearch | $0 (local Docker) |
| Postgres | $0 |
| Phoenix | $0 (self-hosted OSS) |
| HHEM-2.1-open | $0 (CPU inference) |
| Embeddings | $0 (sentence-transformers) |
| Bedrock Haiku 4.5 — full eval | ~$3-5 (auto-tracked by litellm) |
| Optional t3.medium for headless run (2 hrs) | ~$0.10 |
| **Total dissertation eval cost** | **<$10** |

`litellm` writes the per-call cost into each row of `runs.cost_usd`. The `cost_per_correct` column in `metrics` is then `SUM(cost_usd) / SUM(is_correct::int)` per system — directly usable in Chapter 4.

---

## 14. Suggested build order (3-4 focused evenings)

1. **Evening 1:** `docker-compose.yml`, Postgres schema, Alembic migration, `cli.py` skeleton, ingest MultiHop + RAGTruth with correct splits. Verify row counts match published.
2. **Evening 2:** OpenSearch indexer with `BAAI/llm-embedder`. Index MultiHop corpus. Smoke-test retrieval via `curl http://localhost:9200/_search`.
3. **Evening 3:** `tracing.py`, `llm/client.py`, HHEM loader, Systems A and C. Calibration script and ROC plot. Run System A on 10 queries; verify Phoenix shows the trace tree.
4. **Evening 4:** Systems B and D using your prod LangGraph (adapted) with Instructor for typed decisions. Full experiment run. Marimo analysis notebook for the dissertation figures.

After this: iterate on prompts, top_k, and max_steps based on what you see in Phoenix. Don't optimise blind.

---

## 15. One-line health check

```bash
docker compose exec api python -c "
from src.db.session import engine
from src.retrieval.opensearch_client import client
from src.faithfulness.hhem import score
from src.llm.client import generate
from sqlalchemy import text
import httpx
print('db:        ', engine.connect().execute(text('select 1')).scalar())
print('opensearch:', client.info()['version']['number'])
print('phoenix:   ', httpx.get('http://phoenix:6006/healthz').status_code)
print('hhem:      ', score([('a cat sat', 'a feline rested')]))
print('bedrock:   ', generate([{'role':'user','content':'say ok'}])['content'][:20])
"
```

Five lines of green = the entire stack is wired. Move on to ingestion.

---

## 16. What I deliberately skipped (and why)

These look tempting and are wrong for this project:

- **Redis / Celery** — your evaluator is a sequential loop.
- **A separate vector DB** — OpenSearch handles k-NN, matches prod.
- **A React/Streamlit frontend** — Phoenix is your dashboard.
- **Kubernetes** — `docker compose` is the right granularity for a dissertation.
- **Authentication / RBAC** — local-only, single user.
- **A separate HHEM microservice** — adds an HTTP hop for a 0.5-second in-process call.
- **LangSmith** — Phoenix is the self-hosted equivalent, zero credentials.
- **Ragas / DeepEval for retrieval metrics** — LLM-as-judge replaces deterministic arithmetic with non-determinism.
- **Prefect / Airflow / Dagster** — a `for` loop with `rich.progress` is the right tool.
- **Cohere Rerank, Voyage embeddings as the primary retriever** — breaks comparability with the MultiHop-RAG paper baseline. Save them for one ablation table.

---

## 17. What changed from v1 (the changelog)

| v1 | v2 | Lines saved | Why better |
|---|---|---|---|
| Custom Bedrock client | `litellm.completion()` | ~40 | Provider-agnostic, automatic cost tracking, retries built in |
| Custom token + cost tracker | `response._hidden_params["response_cost"]` | ~30 | Maintained model-pricing map, accurate per call |
| Streamlit dashboard container | Phoenix container | ~120 (Streamlit app) | Real OTel tracing, embedding viz, eval logging, dataset versioning |
| Custom trajectory JSONB logging | Phoenix spans + `phoenix_trace_id` pointer | ~50 | Drill-down per query, filter by attributes, no schema-versioning headaches |
| Ad-hoc JSON parsing for agent action | Instructor + typed `AgentDecision` enum | ~25 | Cannot produce a malformed action, ever |
| `pip install` in Dockerfile | `uv pip install` | (faster builds, ~5-10 min saved per rebuild) | 10-30× faster |
| Plain CLI prints | Typer + Rich progress | ~minor | Polished demo, accurate ETA on the 1600-call run |
| Ad-hoc Jupyter for analysis | Marimo notebook | (reproducibility) | Reactive, deterministic, git-friendly |

**Net result:** ~265 fewer lines of bespoke code to write, debug, and defend in the viva. Same architectural shape; far better tooling underneath.

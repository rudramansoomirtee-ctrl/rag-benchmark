"""Phoenix tracing setup. Called once at startup and at the top of each CLI command.

After init_tracing() runs, every LangChain/LangGraph node and every litellm.completion()
call is automatically traced. You don't write tracing code anywhere else.
"""
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

from src.config import settings

_initialized = False


def init_tracing(project_name: str = "rag-benchmark"):
    """Idempotent — safe to call from both FastAPI startup and CLI commands."""
    global _initialized
    if _initialized:
        return None

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{settings.phoenix_collector_endpoint}/v1/traces",     
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    _initialized = True
    return tracer_provider

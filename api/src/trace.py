"""Opt-in glass-box trace recorder for the Ask endpoint.

Building-block functions (retrieve, decompose, route) call `trace_event`; it is a
no-op unless a recorder is active (set by `capture()`). Experiment runs never call
`capture()`, so their behaviour, cost and determinism are completely unaffected —
the controlled comparison is untouched. Only an explicit `POST /api/ask {trace:true}`
turns recording on.
"""
import contextlib
import contextvars

_RECORDER: contextvars.ContextVar = contextvars.ContextVar("glassbox_trace", default=None)


def trace_active() -> bool:
    return _RECORDER.get() is not None


def trace_event(kind: str, **data) -> None:
    rec = _RECORDER.get()
    if rec is not None:
        rec.append({"kind": kind, **data})


@contextlib.contextmanager
def capture():
    """Collect trace events emitted within the block into a fresh list."""
    events: list[dict] = []
    token = _RECORDER.set(events)
    try:
        yield events
    finally:
        _RECORDER.reset(token)

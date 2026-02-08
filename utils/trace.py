from functools import wraps
from contextvars import ContextVar
import os


_trace_depth: ContextVar[int] = ContextVar("_trace_depth", default=0)
_TAG_PREFIXES = {
    "profiling.cv": "CV",
    "profiling.ingestion": "INGEST",
    "profiling.utils": "PROFILING UTILS",
    "utils": "UTILS",
    "profiling.dev": "DEV",
    "profiling.profile_generator": "PROFILE GEN",
    "profiling.nlp": "NLP",
    "profiling.embedding": "EMBEDDING",
}


def _module_tag(module: str) -> str:
    for prefix, tag in _TAG_PREFIXES.items():
        if module.startswith(prefix):
            return tag
    return module.split(".")[0].upper()


def trace(fn=None, *, tag: str | None = None):
    if fn is None:
        return lambda real_fn: trace(real_fn, tag=tag)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if os.getenv("TRACE_ENABLED", "false").lower() != "true":
            return fn(*args, **kwargs)

        trace_cv = os.getenv("TRACE_CV", "false").lower() == "true"
        trace_nlp = os.getenv("TRACE_NLP", "false").lower() == "true"
        trace_embed = os.getenv("TRACE_EMBEDDING", "false").lower() == "true"
        if fn.__module__.startswith("profiling.cv") and not trace_cv:
            return fn(*args, **kwargs)
        if fn.__module__.startswith("profiling.nlp") and not trace_nlp:
            return fn(*args, **kwargs)
        if fn.__module__.startswith("profiling.embedding") and not trace_embed:
            return fn(*args, **kwargs)

        depth = _trace_depth.get()
        indent = "  " * depth
        resolved_tag = tag or _module_tag(fn.__module__)
        print(f"{indent}[TRACE][{resolved_tag}] {fn.__module__}.{fn.__name__}")
        token = _trace_depth.set(depth + 1)
        try:
            return fn(*args, **kwargs)
        finally:
            _trace_depth.reset(token)
    return wrapper

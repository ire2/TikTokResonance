from functools import wraps
from contextvars import ContextVar


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
    "profiling.resonance": "RESONANCE"
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

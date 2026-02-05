from functools import wraps


def trace(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[TRACE] {fn.__module__}.{fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

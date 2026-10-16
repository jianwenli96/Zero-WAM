"""Caches scoped to one serial RTC request, including its denoising iterations."""
from contextlib import contextmanager
from contextvars import ContextVar

_cache = ContextVar('zero_wam_rtc_request_cache', default=None)


@contextmanager
def request_cache():
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def memo(key, compute):
    cache = _cache.get()
    if cache is None or key is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]

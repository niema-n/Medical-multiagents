from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar


logger = logging.getLogger("medical_multiagents.performance")

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def perf_timer(name: str, **metadata: Any) -> Iterator[dict[str, float]]:
    """Log elapsed time for the hot HITL path without changing behavior."""
    started = time.perf_counter()
    stats = {"elapsed_ms": 0.0}
    try:
        yield stats
    finally:
        stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "perf_%s elapsed_ms=%s metadata=%s",
            name,
            stats["elapsed_ms"],
            {key: value for key, value in metadata.items() if value is not None},
        )


def timed(name: str) -> Callable[[F], F]:
    """Decorator variant for small utility functions."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with perf_timer(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def with_retry(
    *,
    retries: int = 3,
    delay_seconds: float = 0.05,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: Exception | None = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as exc:
                    last_exc = exc
                    if attempt == retries - 1:
                        break
                    time.sleep(delay_seconds * (attempt + 1))
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator

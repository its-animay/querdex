from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class StructuredLogger:
    name: str

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def info(self, event: str, **kwargs: Any) -> None:
        payload = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": "info",
            "event": event,
            **kwargs,
        }
        self._logger.info(json.dumps(payload, default=str))

    @contextmanager
    def span(self, event: str, **kwargs: Any) -> Iterator[str]:
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        self.info(f"{event}_start", trace_id=trace_id, **kwargs)
        try:
            yield trace_id
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self.info(f"{event}_end", trace_id=trace_id, duration_ms=duration_ms, **kwargs)

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("medical_multiagents")


@contextmanager
def agent_timer(agent_name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
        logger.info("agent_completed", extra={"agent": agent_name, "duration_ms": int((time.perf_counter() - start) * 1000)})
    except Exception:
        logger.exception("agent_failed", extra={"agent": agent_name, "duration_ms": int((time.perf_counter() - start) * 1000)})
        raise

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


def new_trace_id() -> str:
    # short id for console readability
    return uuid.uuid4().hex[:10]


def _now_ms() -> int:
    return int(time.time() * 1000)


def trace_enabled() -> bool:
    v = (os.getenv("TRACE_ENABLED") or "").strip()
    if v == "":
        return True
    try:
        return bool(int(v))
    except Exception:
        return True


def log_stage(
    *,
    trace_id: str,
    stage: str,
    event: str,
    ms: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not trace_enabled():
        return
    tid = (trace_id or "").strip() or "-"
    st = (stage or "").strip() or "-"
    ev = (event or "").strip() or "-"
    parts = [f"[trace={tid}]", f"[stage={st}]", ev]
    if ms is not None:
        parts.append(f"ms={int(ms)}")
    if extra:
        # keep it compact
        try:
            kv = []
            for k, v in list(extra.items())[:12]:
                s = str(v)
                if len(s) > 120:
                    s = s[:120] + "…"
                kv.append(f"{k}={s}")
            if kv:
                parts.append(" ".join(kv))
        except Exception:
            pass
    print(" ".join(parts))


@dataclass
class StageTimer:
    trace_id: str
    stage: str
    extra_start: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._t0 = _now_ms()
        log_stage(trace_id=self.trace_id, stage=self.stage, event="start", extra=self.extra_start)

    def ok(self, *, extra: Optional[Dict[str, Any]] = None) -> None:
        dt = _now_ms() - self._t0
        log_stage(trace_id=self.trace_id, stage=self.stage, event="ok", ms=dt, extra=extra)

    def fail(self, err: Exception) -> None:
        dt = _now_ms() - self._t0
        log_stage(trace_id=self.trace_id, stage=self.stage, event="fail", ms=dt, extra={"err": str(err)})


@contextmanager
def stage_timer(
    *,
    trace_id: str,
    stage: str,
    extra_start: Optional[Dict[str, Any]] = None,
) -> Iterator[StageTimer]:
    t = StageTimer(trace_id=trace_id, stage=stage, extra_start=extra_start)
    try:
        yield t
        t.ok()
    except Exception as e:
        t.fail(e)
        raise


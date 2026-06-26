"""
latency_tracer.py
=================
Microsecond-level latency instrumentation for the sniping pipeline.

You cannot optimize what you do not measure. This module traces each stage of
a trade's lifecycle and exposes p50/p95/p99 percentiles so we can see exactly
where time is lost.

The giants know their p50 is <5ms because they instrument everything. Most
retail bots log "buy executed" with a timestamp and nothing else. We aim to
be as disciplined as the giants about measurement, even if our absolute
numbers are higher.

Traced stages (each optional; missing stages are skipped in the percentile calc):

  detect        — logsSubscribe event received
  parsed        — mint extracted from the event
  scored        — anti-rug + token_scorer finished
  built         — pump.fun instruction built
  submitted     — tx sent to RPC / Jito
  confirmed     — tx landed on-chain

For a snipe, the headline metric is detect→confirmed (total round-trip).
For diagnosis, the per-stage breakdown shows where the time goes.

Usage:
    tracer = LatencyTracer()
    trace_id = tracer.start("snipe", token="EPjF...")
    tracer.mark(trace_id, "parsed")
    ...
    tracer.mark(trace_id, "confirmed")
    tracer.finish(trace_id)
    stats = tracer.stats()  # {snipe: {total_ms: {p50,p95,p99}, stages: {...}}}
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import setup_logger

log = setup_logger("latency")

# Max traces to keep in memory for percentile calculation.
_MAX_TRACES_PER_OP = 500


@dataclass
class Trace:
    op: str                  # "snipe" | "buy" | "sell" | "exit"
    token: str
    start: float
    marks: dict[str, float] = field(default_factory=dict)
    finished: bool = False


class LatencyTracer:
    """
    Thread-safe-ish (asyncio single-thread) latency tracer.
    All times are in milliseconds, measured with time.perf_counter (monotonic).
    """

    def __init__(self) -> None:
        # op -> list of (total_ms, {stage: ms_from_start})
        self._completed: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_MAX_TRACES_PER_OP)
        )
        self._active: dict[int, Trace] = {}
        self._next_id: int = 0
        # Rolling alert thresholds (ms). If a total exceeds these, log.warning.
        self.alert_thresholds: dict[str, float] = {
            "snipe": 2000.0,
            "buy": 5000.0,
            "sell": 5000.0,
            "exit": 3000.0,
        }

    def start(self, op: str, token: str = "") -> int:
        """Begin a trace. Returns a trace_id to pass to mark()/finish()."""
        tid = self._next_id
        self._next_id += 1
        self._active[tid] = Trace(op=op, token=token, start=time.perf_counter())
        return tid

    def mark(self, trace_id: int, stage: str) -> None:
        """Record that `stage` completed for this trace."""
        tr = self._active.get(trace_id)
        if tr is None:
            return
        tr.marks[stage] = (time.perf_counter() - tr.start) * 1000.0

    def finish(self, trace_id: int) -> Optional[float]:
        """
        Close the trace and record it. Returns total_ms, or None if unknown id.
        """
        tr = self._active.pop(trace_id, None)
        if tr is None:
            return None
        total_ms = (time.perf_counter() - tr.start) * 1000.0
        # Store as (total, {stage: relative_ms})
        self._completed[tr.op].append((total_ms, dict(tr.marks)))

        # Alert on slow operations
        threshold = self.alert_thresholds.get(tr.op, 999999.0)
        if total_ms > threshold:
            log.warning("latency.slow_op",
                        op=tr.op, token=tr.token,
                        total_ms=round(total_ms, 1),
                        threshold=threshold,
                        stages={k: round(v, 1) for k, v in tr.marks.items()})
        else:
            log.info("latency.trace",
                     op=tr.op, token=tr.token,
                     total_ms=round(total_ms, 1),
                     stages={k: round(v, 1) for k, v in tr.marks.items()})
        return total_ms

    def stats(self) -> dict:
        """
        Return percentile stats per operation.
        {
          "snipe": {
            "count": 142,
            "total_ms": {"p50": 612, "p95": 1402, "p99": 2103},
            "stages": {
              "detect->parsed": {"p50": 45, "p95": 120, "p99": 200},
              "parsed->scored": {"p50": 180, "p95": 450, "p99": 800},
              ...
            }
          }
        }
        """
        result: dict = {}
        for op, traces in self._completed.items():
            if not traces:
                continue
            totals = [t[0] for t in traces]
            entry = {
                "count": len(totals),
                "total_ms": self._percentiles(totals),
                "stages": {},
            }
            # Compute inter-stage deltas. We need pairs of marks in order.
            # Collect all stage names seen, then for consecutive pairs compute
            # the delta distribution.
            stage_names: list[str] = []
            seen: set[str] = set()
            for _, marks in traces:
                for s in marks:
                    if s not in seen:
                        seen.add(s)
                        stage_names.append(s)
            for i in range(len(stage_names) - 1):
                a, b = stage_names[i], stage_names[i + 1]
                deltas = []
                for _, marks in traces:
                    if a in marks and b in marks:
                        deltas.append(marks[b] - marks[a])
                if deltas:
                    entry["stages"][f"{a}->{b}"] = self._percentiles(deltas)
            # Also first-stage (start -> first mark) if present
            if stage_names:
                first = stage_names[0]
                deltas = [marks[first] for _, marks in traces if first in marks]
                if deltas:
                    entry["stages"][f"start->{first}"] = self._percentiles(deltas)
            result[op] = entry
        return result

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0}
        s = sorted(values)
        n = len(s)
        def pct(p: float) -> float:
            idx = max(0, min(n - 1, int(p / 100 * (n - 1))))
            return round(s[idx], 1)
        return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}


# Module-level singleton (the orchestrator and strategies share one instance)
_tracer: Optional[LatencyTracer] = None


def get_tracer() -> LatencyTracer:
    global _tracer
    if _tracer is None:
        _tracer = LatencyTracer()
    return _tracer

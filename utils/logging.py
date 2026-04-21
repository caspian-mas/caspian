"""
caspian/utils/logging.py

Structured logger for CASPIAN.

Emits events that map to the algorithm's key steps:
  - cascade onset fired   (step 15: Z(t0) > tau)
  - cascade window closed (step 19: [t0, t1] finalised)
  - attribution complete  (steps 21-23: origin, bridge, amplifier)
  - turn metrics          (step 12/13: per-turn λ1, r, p, Z)
  - benchmark run start/end

Two output modes:
  plain  — human-readable, default for development
  json   — one JSON object per line, for log aggregation / CI

All structured fields use the exact variable names from the spec
(t0, t1, Z, tau, lambda1, r, p, origin, bridge, amplifiers)
so log output is directly traceable to algorithm steps.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_LOGGER_NAME = "caspian"


# ── Public setup ───────────────────────────────────────────────────────────

def configure_logging(
    level       : str  = "INFO",
    json_output : bool = False,
) -> None:
    """
    Call once at detector init. Idempotent — safe to call multiple times.

    Parameters
    ----------
    level       : stdlib level string ("DEBUG", "INFO", "WARNING", "ERROR")
    json_output : if True, emit one JSON object per line instead of plain text
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Don't add duplicate handlers on repeated calls
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = (
        _JsonFormatter() if json_output
        else logging.Formatter(
            fmt     = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt = "%H:%M:%S",
        )
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


# ── Algorithm-step events ──────────────────────────────────────────────────

def log_turn_metrics(
    turn    : int,
    lambda1 : float,
    r       : float,
    p       : float,
    z_lambda: float,
    z_r     : float,
    z_p     : float,
    Z       : float,
) -> None:
    """
    Steps 12/13: per-turn metrics and normalised anomaly scores.
    Logged at DEBUG to avoid noise in normal runs.
    """
    get_logger().debug(
        "turn metrics",
        extra={
            "event":    "turn_metrics",
            "turn":     turn,
            "lambda1":  round(lambda1, 6),
            "r":        round(r,       6),
            "p":        round(p,       6),
            "z_lambda": round(z_lambda,4),
            "z_r":      round(z_r,     4),
            "z_p":      round(z_p,     4),
            "Z":        round(Z,       4),
        },
    )


def log_onset(turn: int, Z: float, tau: float) -> None:
    """
    Step 15: cascade onset declared — Z(t0) > tau.
    """
    get_logger().info(
        f"Cascade onset at turn {turn}  Z={Z:.4f}  tau={tau}",
        extra={
            "event": "onset",
            "turn":  turn,
            "Z":     round(Z,   4),
            "tau":   tau,
        },
    )


def log_cascade_window(t0: int, t1: int, cascade_type: str) -> None:
    """
    Step 19: cascade window [t0, t1] finalised and type classified.
    """
    get_logger().info(
        f"Cascade window [{t0}, {t1}]  type={cascade_type}",
        extra={
            "event":        "cascade_window",
            "t0":           t0,
            "t1":           t1,
            "cascade_type": cascade_type,
        },
    )


def log_attribution(
    origin     : str,
    bridge     : str | None,
    amplifiers : list[str],
) -> None:
    """
    Steps 21-23: attribution inference complete.
    """
    get_logger().info(
        f"Attribution — origin={origin}  bridge={bridge}  "
        f"amplifiers={amplifiers}",
        extra={
            "event":      "attribution",
            "origin":     origin,
            "bridge":     bridge,
            "amplifiers": amplifiers,
        },
    )


def log_spines(spines: list) -> None:
    """
    Step 24: top-K spines ranked.
    Logs the top spine path and score at INFO, rest at DEBUG.
    """
    logger = get_logger()
    for k, spine in enumerate(spines):
        msg = f"Spine {k+1}: {' -> '.join(spine.nodes)}  score={spine.score:.4f}"
        if k == 0:
            logger.info(msg, extra={"event": "spine", "rank": k+1,
                                    "path": spine.nodes, "score": spine.score})
        else:
            logger.debug(msg, extra={"event": "spine", "rank": k+1,
                                     "path": spine.nodes, "score": spine.score})


def log_benchmark_start(benchmark: str, n_scenarios: int) -> None:
    get_logger().info(
        f"Benchmark start  name={benchmark}  scenarios={n_scenarios}",
        extra={"event": "benchmark_start", "benchmark": benchmark,
               "n_scenarios": n_scenarios},
    )


def log_benchmark_end(
    benchmark  : str,
    precision  : float,
    recall     : float,
    f1         : float,
    elapsed    : float,
) -> None:
    get_logger().info(
        f"Benchmark end  name={benchmark}  "
        f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
        f"elapsed={elapsed:.2f}s",
        extra={
            "event":     "benchmark_end",
            "benchmark": benchmark,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "elapsed":   round(elapsed,   3),
        },
    )


# ── JSON formatter ─────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    Emits one JSON object per line.
    Structured `extra` fields are merged into the top-level object.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Merge any structured extra fields
        for key, val in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTIN_ATTRS:
                payload[key] = val
        return json.dumps(payload, default=str)


# stdlib LogRecord attributes to exclude from JSON output
_LOG_RECORD_BUILTIN_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})
"""
caspian/utils/rolling.py

Rolling median and MAD for online self-normalization (step 13).

Spec (page 6):
    med_W(q, t)  = rolling median over last W observations of q
    MAD_W(q, t)  = rolling median absolute deviation over last W observations
    z̃_q(t)      = (q(t) - med_W(q, t)) / (MAD_W(q, t) + ε)

The window is PAST-EXCLUSIVE: med and MAD are computed from the W
observations BEFORE the current value, then the current value is scored
against that baseline. This matches the spec's intent — self-normalization
against the agent's own recent history, not against itself.
"""

from __future__ import annotations

from collections import deque
import numpy as np


class RollingStats:
    """
    Tracks a fixed-size window of past observations and scores
    a new value against that window's median and MAD.

    Parameters
    ----------
    window : int
        W — number of past turns to keep (must be >= 2)
    eps : float
        ε — MAD stability denominator (from DetectorConfig.eps_mad)
    """

    def __init__(self, window: int, eps: float = 1e-8) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.eps    = eps
        self._buf : deque[float] = deque(maxlen=window)

    # ── Core API ───────────────────────────────────────────────────────────

    def score(self, value: float) -> float:
        """
        Score `value` against the current window (PAST observations only),
        then push `value` into the window for future turns.

        Returns z̃_q(t) = (value - med_W) / (MAD_W + ε).
        Returns 0.0 until the window has >= 2 observations.
        """
        z = self._compute_z(value)
        self._buf.append(float(value))
        return z

    def update(self, value: float) -> None:
        """Push a value into the window without scoring (warm-up use)."""
        self._buf.append(float(value))

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True once the window holds >= 2 observations."""
        return len(self._buf) >= 2

    @property
    def median(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.median(list(self._buf)))

    @property
    def mad(self) -> float:
        """Median absolute deviation of current window."""
        if len(self._buf) < 2:
            return 0.0
        arr = np.array(self._buf, dtype=np.float64)
        return float(np.median(np.abs(arr - np.median(arr))))

    @property
    def n(self) -> int:
        return len(self._buf)

    # ── Internal ───────────────────────────────────────────────────────────

    def _compute_z(self, value: float) -> float:
        """
        Compute z̃ using current window state (before pushing value).
        Returns 0.0 if window not yet ready.
        """
        if not self.ready:
            return 0.0
        med = self.median
        mad = self.mad
        return (value - med) / (mad + self.eps)

    def reset(self) -> None:
        self._buf.clear()


class MultiMetricRolling:
    """
    One RollingStats per metric. Scores all metrics in one call.

    Used by core/metrics.py to track λ1, r, p simultaneously and
    return z̃_λ, z̃_r, z̃_p each turn.

    Parameters
    ----------
    metric_names : list of str
        e.g. ["lambda1", "r", "p"]
    window : int
        W from DetectorConfig
    eps : float
        ε from DetectorConfig.eps_mad
    """

    def __init__(
        self,
        metric_names : list[str],
        window       : int,
        eps          : float = 1e-8,
    ) -> None:
        self._stats: dict[str, RollingStats] = {
            name: RollingStats(window, eps) for name in metric_names
        }

    def score(self, values: dict[str, float]) -> dict[str, float]:
        """
        Score each metric against its past window, then update windows.
        Keys in `values` must match metric_names.

        Returns dict of z̃ values keyed by metric name.
        """
        return {
            name: self._stats[name].score(v)
            for name, v in values.items()
        }

    @property
    def ready(self) -> bool:
        """True once ALL metric windows have >= 2 observations."""
        return all(s.ready for s in self._stats.values())

    def reset(self) -> None:
        for s in self._stats.values():
            s.reset()

    def get(self, name: str) -> RollingStats:
        return self._stats[name]
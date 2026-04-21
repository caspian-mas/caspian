"""
caspian/core/onset.py

Steps 15, 18, 19 — Cascade onset detection, cascade window management,
and single-step vs multi-step classification.

Step 15  Onset condition:
             Declare onset at first turn t0 where Z(t0) > τ
             τ is a monitor operating point — calibrate on held-out runs.

Step 18  Classification:
    18.1 Single-step: Z spikes sharply, slope_W(λ1, t0) is large,
                      AND channel mass peaks are synchronous
                      (peak times differ by at most 1 turn around onset).
    18.2 Multi-step:  Z stays elevated over multiple turns,
                      slope_W(λ1,t), slope_W(r,t), or slope_W(p,t)
                      remain positive over a window,
                      AND channel peaks are phase-shifted
                      (differ by more than 1 turn),
                      AND anomaly persists across persistence horizon mp.
    Note: mp is a labeling horizon — not a hidden system truth.

Step 19  Cascade window [t0, t1]:
             t0 = onset turn (first Z(t) > τ)
             t1 = last turn in anomalous segment, extended while Z > τ.
             Window closes mp turns after Z last exceeded τ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from utils.types import Channel, CascadeType, TurnMetrics
from utils.config import DetectorConfig
from core.metrics import MetricsEngine


@dataclass
class CascadeWindow:
    """
    The active cascade interval [t0, t1].

    t0     : onset turn — first Z(t0) > τ
    t1     : last turn where Z > τ (extended each turn while elevated)
    active : True until window is closed (mp turns of Z <= τ elapsed)
    """
    t0     : int
    t1     : int
    active : bool = True

    def extend(self, turn: int) -> None:
        self.t1 = turn

    def close(self, turn: int) -> None:
        self.t1    = turn
        self.active = False


class OnsetDetector:
    """
    Stateful onset detector. Receives one TurnMetrics per turn,
    manages the cascade window, and classifies cascade type on close.

    Parameters
    ----------
    config        : DetectorConfig — τ, W, mp all sourced from here
    metrics_engine: MetricsEngine  — provides windowed slopes and
                                     channel mass histories for classification
    """

    def __init__(
        self,
        config         : DetectorConfig,
        metrics_engine : MetricsEngine,
    ) -> None:
        self._tau     = config.tau
        self._W       = config.W
        self._mp      = config.mp
        self._metrics = metrics_engine

        # Active cascade window (None when no cascade in progress)
        self._window  : CascadeWindow | None = None

        # Turns since Z last exceeded τ (used for mp persistence check)
        self._turns_below_tau : int = 0

        # Completed windows emitted this run
        self._completed : list[CascadeWindow] = []

        # Snapshot of metric state taken AT onset (t0) — used for
        # classification so we capture burst-vs-drift at the moment
        # it fires, not at window close time.
        self._onset_snapshot: dict = {}

    # ── Main per-turn update ───────────────────────────────────────────────

    def update(self, metrics: TurnMetrics) -> tuple[bool, CascadeWindow | None]:
        """
        Process one turn's TurnMetrics.

        Returns
        -------
        (onset_fired, closed_window)
            onset_fired    : True if this is the turn Z(t) first exceeded τ
            closed_window  : CascadeWindow if a window just closed, else None
        """
        turn        = metrics.turn
        Z           = metrics.Z
        onset_fired = False
        closed_window: CascadeWindow | None = None

        if Z > self._tau:
            self._turns_below_tau = 0

            if self._window is None:
                # Step 15: declare onset at first Z(t0) > τ
                self._window = CascadeWindow(t0=turn, t1=turn, active=True)
                onset_fired  = True
                # Snapshot onset-local state for classification
                # Classification must use features AT t0, not at window close
                self._onset_snapshot = {
                    "rise_Z":   self._metrics.windowed_rise("lambda1"),
                    "slope_l1": self._metrics.windowed_slope("lambda1"),
                    "slope_r":  self._metrics.windowed_slope("r"),
                    "slope_p":  self._metrics.windowed_slope("p"),
                    "mass_histories": self._metrics.all_channel_mass_histories(),
                    "Z":        metrics.Z,
                    "turn":     turn,
                }
            else:
                # Extend existing window
                self._window.extend(turn)

        else:
            # Z <= τ this turn
            if self._window is not None and self._window.active:
                self._turns_below_tau += 1

                # Step 19: close window after mp consecutive turns below τ
                if self._turns_below_tau >= self._mp:
                    self._window.close(turn)
                    self._completed.append(self._window)
                    closed_window = self._window
                    self._window  = None
                    self._turns_below_tau = 0

        return onset_fired, closed_window

    # ── Step 18: cascade type classification ──────────────────────────────

    def classify(self, window: CascadeWindow) -> CascadeType:
        """
        Classify a closed cascade window as SINGLE_STEP or MULTI_STEP.

        IMPORTANT: classification uses the onset-local snapshot taken
        AT t0, not the rolling state at window-close time. This is
        because burst vs drift is determined by what happens at onset:
            - a burst hits all channels simultaneously at t0
            - a drift builds gradually across turns

        Step 18.1 Single-step:
            1. Z spiked sharply at onset — large windowed rise at t0
            2. slope_W(λ1, t0) is large — eigenvalue was growing
            3. Channel mass peaks synchronous at t0 — peaks within 1 turn

        Step 18.2 Multi-step:
            1. Z stayed elevated (duration >= mp)
            2. At least one slope positive at onset
            3. Channel mass peaks phase-shifted — differ by > 1 turn

        Note: mp is a labeling horizon, not a hidden system truth.
        Default is MULTI_STEP if conditions are ambiguous.
        """
        snap = self._onset_snapshot

        # Use onset-local values from snapshot
        rise_Z   = snap.get("rise_Z",   0.0)
        slope_l1 = snap.get("slope_l1", 0.0)
        slope_r  = snap.get("slope_r",  0.0)
        slope_p  = snap.get("slope_p",  0.0)

        # Channel mass peak spread using onset-local mass histories
        mass_histories = snap.get("mass_histories", {})
        peak_spread    = self._channel_peak_spread_from(mass_histories)

        # Cascade duration
        duration = window.t1 - window.t0

        # ── Single-step check (all three conditions must hold) ─────────────
        sharp_spike = rise_Z   > 0.5    # Z rose sharply at onset
        large_slope = slope_l1 > 0.0    # λ1 was growing at onset
        synchronous = peak_spread <= 1  # channel peaks within 1 turn

        if sharp_spike and large_slope and synchronous:
            return CascadeType.SINGLE_STEP

        # ── Multi-step check ───────────────────────────────────────────────
        slopes_positive = slope_l1 > 0.0 or slope_r > 0.0 or slope_p > 0.0
        phase_shifted   = peak_spread > 1
        persistent      = duration >= self._mp

        if slopes_positive and phase_shifted and persistent:
            return CascadeType.MULTI_STEP

        # Default: multi-step (conservative)
        return CascadeType.MULTI_STEP

    def _channel_peak_spread_from(
        self,
        mass_histories: dict,
    ) -> int:
        """
        Compute spread (in turns) between earliest and latest channel
        mass peak across the four M_c series.

        Uses the onset-local snapshot of mass histories so classification
        reflects channel timing AT onset, not at window close.

        Returns max_peak_idx - min_peak_idx over the last W entries.
        Returns 0 if fewer than 2 channels have data.
        """
        peak_positions: list[int] = []

        for channel in Channel:
            hist = mass_histories.get(channel, [])
            if not hist:
                continue
            window_hist = hist[-self._W:]
            if not window_hist:
                continue
            peak_positions.append(int(np.argmax(window_hist)))

        if len(peak_positions) < 2:
            return 0

        return max(peak_positions) - min(peak_positions)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def current_window(self) -> CascadeWindow | None:
        """Active cascade window, or None if no cascade in progress."""
        return self._window

    @property
    def in_cascade(self) -> bool:
        return self._window is not None and self._window.active

    @property
    def completed_windows(self) -> list[CascadeWindow]:
        return list(self._completed)

    def reset(self) -> None:
        """Clear all state. Call between benchmark scenarios."""
        self._window            = None
        self._turns_below_tau   = 0
        self._completed.clear()
        self._onset_snapshot    = {}
"""
caspian/core/metrics.py

Steps 12-17 — Direct detection metrics, self-normalization, onset
score, windowed trends, and channel mass time series.

Step 12.1  λ1(t) = λ_max(A_t)                        [Perron root]
Step 12.2  r(t)  = λ2(t) / (λ1(t) + ε)               [synchrony ratio]
Step 12.3  p(t)  = cross-channel 2-hop proportion     [channel pivot metric]
               p(t) = #{(i,j),(j,k) ∈ E_act : χ(i,j) ≠ χ(j,k)}
                      / (#{(i,j),(j,k) ∈ E_act} + ε)
               Uses ALL active edges — not an arbitrary top fraction.

Step 13    z̃_q(t) = (q(t) - med_W(q,t)) / (MAD_W(q,t) + ε)
               Local self-normalization. No global baseline assumed.
               Window is PAST-EXCLUSIVE: scored against prior W turns.

Step 14    Z(t) = min(z̃_λ, z̃_r, z̃_p)
               Weakest-link: ALL three signals must be elevated.

Step 16.1  Δ_W q(t) = q(t) - q(t - W + 1)            [windowed rise]
Step 16.2  slope_W(q, t) = least-squares slope over   [windowed slope]
               {(u, q(u)) : u = t-W+1,...,t}

Step 17    M_c(t) = Σ_{i,j} A_t^(c)[i,j]             [channel mass]
               Four time series — used for single/multi-step classification.

Output: TurnMetrics dataclass consumed by core/onset.py
"""

from __future__ import annotations

import numpy as np

from utils.types import Channel, TurnMetrics
from utils.config import DetectorConfig
from utils.rolling import MultiMetricRolling
from utils.graph_utils import top_two_eigenvalues
from utils.config import SmallScaleConfig, MediumScaleConfig, LargeScaleConfig
from core.channel_matrix import ChannelMatrices


class MetricsEngine:
    """
    Stateful metrics engine. Maintains rolling windows across turns
    and produces one TurnMetrics per turn.

    Also maintains per-metric history lists for windowed trend
    computation (steps 16.1, 16.2) used by core/onset.py for
    single-step vs multi-step classification.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config  = config
        self._W       = config.W
        self._eps_mad = config.eps_mad
        self._eps_act = config.eps_active

        # Step 13: rolling windows for the three metrics
        self._rolling = MultiMetricRolling(
            metric_names = ["lambda1", "r", "p"],
            window       = config.W,
            eps          = config.eps_mad,
        )

        # History lists for windowed trend computation (step 16)
        # Kept at W+1 entries max
        self._lambda1_hist : list[float] = []
        self._r_hist       : list[float] = []
        self._p_hist       : list[float] = []

        # Channel mass histories (step 17) — one list per channel
        self._mass_hist: dict[Channel, list[float]] = {
            c: [] for c in Channel
        }

    # ── Main per-turn computation ──────────────────────────────────────────

    def compute(self, cm: ChannelMatrices, turn: int) -> TurnMetrics:
        """
        Compute all metrics for turn t from the fused matrix A_t
        and channel matrices inside cm.

        Returns TurnMetrics with raw values, normalized scores, and Z(t).
        """
        A_t = cm.A_t

        # ── Step 12.1: λ1(t) = λ_max(A_t) ────────────────────────────────
        method, kwargs = self._eigen_method()
        lambda1, lambda2 = top_two_eigenvalues(A_t, method=method, **kwargs)

        # ── Step 12.2: r(t) = λ2(t) / (λ1(t) + ε) ───────────────────────
        r = lambda2 / (lambda1 + self._config.eps_norm)  # eps_norm stabilises eigenvalue ratio

        # ── Step 12.3: p(t) cross-channel propagation ─────────────────────
        p = self._cross_channel_proportion(cm)

        # ── Step 13: robust self-normalization ────────────────────────────
        # score() is past-exclusive: scored against prior W turns,
        # then current value pushed into window
        scores   = self._rolling.score({"lambda1": lambda1, "r": r, "p": p})
        z_lambda = scores["lambda1"]
        z_r      = scores["r"]
        z_p      = scores["p"]

        # ── Step 14: weakest-link onset score ─────────────────────────────
        Z = min(z_lambda, z_r, z_p)

        # ── Step 17: channel mass ─────────────────────────────────────────
        for c in Channel:
            mass = cm.channel_mass(c)
            self._mass_hist[c].append(mass)
            if len(self._mass_hist[c]) > self._W + 1:
                self._mass_hist[c].pop(0)

        # Update metric histories for windowed trends (step 16)
        self._lambda1_hist.append(lambda1)
        self._r_hist.append(r)
        self._p_hist.append(p)
        for hist in [self._lambda1_hist, self._r_hist, self._p_hist]:
            if len(hist) > self._W + 1:
                hist.pop(0)

        return TurnMetrics(
            turn     = turn,
            lambda1  = lambda1,
            r        = r,
            p        = p,
            z_lambda = z_lambda,
            z_r      = z_r,
            z_p      = z_p,
            Z        = Z,
        )

    # ── Step 12.3: cross-channel propagation ──────────────────────────────

    def _cross_channel_proportion(self, cm: ChannelMatrices) -> float:
        """
        p(t) = #{(i,j),(j,k) ∈ E_act : χ_t(i,j) ≠ χ_t(j,k)}
               / (#{(i,j),(j,k) ∈ E_act} + ε)

        Spec: use ALL active edges — not an arbitrary top fraction.
        E_act = { (i,j) : A_t[i,j] > ε_A }

        A 2-hop path (i→j→k) counts toward the numerator when the
        dominant channel changes at the pivot node j.
        """
        active = cm.active_edges(self._eps_act)
        if not active:
            return 0.0

        # Build active edge set for O(1) lookup
        # Build per-node adjacency for O(N·k²) instead of O(|E_act|²)
        # in_edges[mid]  = list of src nodes pointing to mid
        # out_edges[mid] = list of tgt nodes mid points to
        in_edges:  dict[str, list[str]] = {}
        out_edges: dict[str, list[str]] = {}
        for (src, tgt) in active:
            out_edges.setdefault(src, []).append(tgt)
            in_edges.setdefault(tgt,  []).append(src)

        total_paths      = 0
        cross_chan_paths = 0

        # Enumerate 2-hop paths via pivot node mid
        for mid in set(in_edges) & set(out_edges):
            for src in in_edges[mid]:
                for tgt in out_edges[mid]:
                    total_paths += 1
                    c_in  = cm.dominant_channel(src, mid)
                    c_out = cm.dominant_channel(mid, tgt)
                    if c_in != c_out:
                        cross_chan_paths += 1

        return cross_chan_paths / (total_paths + self._eps_mad)

    # ── Steps 16.1, 16.2: windowed trends ─────────────────────────────────

    def windowed_rise(self, metric: str) -> float:
        """
        Step 16.1: Δ_W q(t) = q(t) - q(t - W + 1)

        Returns 0.0 if fewer than W observations available.
        """
        hist = self._get_hist(metric)
        if len(hist) < self._W:
            return 0.0
        return hist[-1] - hist[-self._W]

    def windowed_slope(self, metric: str) -> float:
        """
        Step 16.2: Fit least-squares line to (u, q(u)) for
        u = t-W+1,...,t. Returns the slope coefficient.

        Spec: "This is better than using only q(t) - q(t-1)."
        Returns 0.0 if fewer than 2 observations available.
        """
        hist = self._get_hist(metric)
        window = hist[-self._W:]
        n = len(window)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(window, dtype=np.float64)
        # Least-squares slope via the normal equations
        x_mean = x.mean()
        y_mean = y.mean()
        num = float(np.dot(x - x_mean, y - y_mean))
        den = float(np.dot(x - x_mean, x - x_mean))
        if den < 1e-12:
            return 0.0
        return num / den

    def channel_mass_history(self, channel: Channel) -> list[float]:
        """
        M_c history for classification — returns copy of stored series.
        """
        return list(self._mass_hist[channel])

    def all_channel_mass_histories(self) -> dict[Channel, list[float]]:
        """All four M_c time series."""
        return {c: list(v) for c, v in self._mass_hist.items()}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_hist(self, metric: str) -> list[float]:
        return {
            "lambda1": self._lambda1_hist,
            "r":       self._r_hist,
            "p":       self._p_hist,
        }[metric]

    def _eigen_method(self) -> tuple[str, dict]:
        """
        Select eigenvalue method from active scale config.
        SMALL  → dense  (exact np.linalg.eigvals)
        MEDIUM → sparse (ARPACK, k=eigen_k)
        LARGE  → power  (power iteration, λ2=0)
        """
        sc = self._config.active_scale_config
        if isinstance(sc, SmallScaleConfig):
            return "dense", {}
        elif isinstance(sc, MediumScaleConfig):
            return "sparse", {"sparse_k": sc.eigen_k}
        elif isinstance(sc, LargeScaleConfig):
            # λ1 via power iteration; λ2 not computed → r(t) is conservative.
            # Paper note: for large-scale systems r(t) understates synchrony —
            # onset detection relies more heavily on λ1 and p(t) at this scale.
            return "power", {"power_steps": sc.power_iter_steps}
        return "dense", {}

    def reset(self) -> None:
        """Clear all state. Call between benchmark scenarios."""
        self._rolling.reset()
        self._lambda1_hist.clear()
        self._r_hist.clear()
        self._p_hist.clear()
        for v in self._mass_hist.values():
            v.clear()
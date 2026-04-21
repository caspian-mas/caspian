"""
caspian/core/li_cte.py

Steps 6 and 7 — Late-Interaction Conditional Transfer Entropy (LI-CTE).

Spec step 6:
    CTE_ij^(c)(t, τ) = I( U_ij^(c)(t-τ) ; V_ij^(c)(t) | H_j^(c)(t-1) )
    LI-CTE_ij^(c)(t) = max_{τ ∈ {0,...,L}} CTE_ij^(c)(t, τ)

    V  — always current turn t
    U  — lagged by τ turns
    H  — always t-1 (target's own prior, not τ-dependent)
    L  — monitor horizon choice, not a system truth

─────────────────────────────────────────────────────────────────────────
ACTIVE EDGE GRAPH
─────────────────────────────────────────────────────────────────────────
LI-CTE is defined over interaction-restricted state (spec step 5):

    U_ij^(c)(t) ⊆ S_i^(c)(t)
    V_ij^(c)(t) ⊆ S_j^(c)(t)

If there is no interaction support on (i,j,c) at turn t then U=V=∅
and LI-CTE_ij^(c)(t) = 0 by definition. We therefore only compute
over the active interaction graph:

    E_t^act = { (i,j,c) : support pushed this turn with non-empty U or V }

This is NOT an approximation — it is the correct definition.

Complexity: O(|E_t^act| × L × d) ≈ O(N·k·L·d)
where k = avg interaction degree (bounded by system design, k << N).

Runtime is millisecond-scale in optimized implementations for typical
MAS sizes — exact figures depend on Python overhead, batching, and
observed sparsity.

IMPORTANT: The active-edge restriction is only valid if support
extraction captures delayed effects correctly. Memory poisoning must
surface in current-turn V when retrieval happens; delayed communication
uptake must appear in current-turn V even if the message was sent
earlier. The support builder (core/channel_states.py) is responsible
for this — not this file.

─────────────────────────────────────────────────────────────────────────
ESTIMATOR: Linear Residual Transfer Score (LRTS)
─────────────────────────────────────────────────────────────────────────
We compute a residual-based conditional dependence measure over
late-interaction states. Under a Gaussian model this corresponds to
conditional mutual information and preserves relative influence ordering.

    ε_V = V(t) - proj(V(t) onto H(t-1))
    score = cos²(U(t-τ), ε_V)   =  partial R²(U → V | H)

Under Gaussian assumptions: I(U;V|H) = -½ log(1 - R²)
LRTS is a monotone function of Gaussian CMI — correct for detection.

O(d) per edge per channel per lag — no matrix decomposition, no sampling.

Optimizations:
    1. Active-edge restriction  — only edges with interaction support
    2. Channel gating           — skip if both U and V are empty
    3. Zero-residual pruning    — skip if ||ε_V|| ≈ 0
    4. Early lag pruning        — stop if score saturates at any lag
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from utils.types import Channel, LateInteractionSupport
from utils.config import DetectorConfig
from core.channel_states import encode_payload, ENCODE_DIM


# ── LRTS core — O(d) per edge per lag ─────────────────────────────────────

def _lrts(
    U   : np.ndarray,   # (d,)  source state at t-τ
    V   : np.ndarray,   # (d,)  target state at t
    H   : np.ndarray,   # (d,)  target history at t-1
    eps : float = 1e-8,
) -> float:
    """
    Linear Residual Transfer Score:

        ε_V = V - (V·H / ||H||²) * H
        score = clip(cos(U, ε_V), -1, 1)²

    Returns partial R²(U → V | H) ∈ [0, 1].
    Returns 0.0 immediately on zero-residual or zero-source.
    O(3d) scalar ops.
    """
    # Compute residual ε_V = V - proj_H(V)
    h_norm_sq = float(np.dot(H, H))
    if h_norm_sq > eps:
        eps_V = V - (float(np.dot(V, H)) / h_norm_sq) * H
    else:
        eps_V = V   # no history yet — full V is the residual

    # Zero-residual pruning: H fully explains V → no external influence
    ev_norm = float(np.linalg.norm(eps_V))
    if ev_norm < eps:
        return 0.0

    # Zero-source pruning: no signal from U
    u_norm = float(np.linalg.norm(U))
    if u_norm < eps:
        return 0.0

    cos_sim = float(np.dot(U, eps_V)) / (u_norm * ev_norm)
    cos_sim = max(-1.0, min(1.0, cos_sim))   # numerical safety
    return cos_sim ** 2


# ── LI-CTE Engine ──────────────────────────────────────────────────────────

class LICTEEngine:
    """
    Computes LI-CTE_ij^(c)(t) for all active edges each turn.

    Only (src, tgt, channel) triples with pushed interaction support
    are computed — the active-edge graph restriction.

    Usage
    -----
        engine = build_li_cte_engine(config, n_agents)

        # Each turn, for each channel:
        engine.push_supports(supports, channel, turn)

        # Compute over active edges, intersected with E0:
        li_cte = engine.compute_all(graph.feasible_edges_named(), turn)
        # dict[(src, tgt, channel)] -> float
    """

    _EARLY_STOP = 0.95   # lag pruning threshold

    def __init__(
        self,
        config     : DetectorConfig,
        n_agents   : int,
        encode_dim : int = ENCODE_DIM,
    ) -> None:
        self._L          = config.L
        self._encode_dim = encode_dim
        self._exact_cmi  = getattr(config, "exact_cmi", False)
        self._eps        = config.eps_norm

        # Lazy-init exact estimator only when needed
        self._estimator = None
        if self._exact_cmi:
            from caspian.utils.estimator_select import get_estimator
            self._estimator = get_estimator(config)

        # Lag buffer: (src, tgt, channel) ->
        #     list of (turn, U, V, H) oldest-first, capped at L+1
        self._buffer: dict[
            tuple[str, str, Channel],
            list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
        ] = defaultdict(list)

        # Active set for current turn — cleared after compute_all
        self._active: set[tuple[str, str, Channel]] = set()

        # Feasible edge set — set once at init for intersection check
        self._feasible: set[tuple[str, str]] = set()

    def set_feasible_edges(self, edges: list[tuple[str, str]]) -> None:
        """
        Register the feasible edge set E0 from StructuralGraph.
        Called once at detector init. Used to gate active edges
        so we never compute outside E0.
        """
        self._feasible = set(edges)

    # ── Public API ─────────────────────────────────────────────────────────

    def push_supports(
        self,
        supports : dict[tuple[str, str], LateInteractionSupport],
        channel  : Channel,
        turn     : int,
    ) -> None:
        """
        Encode and buffer (U, V, H) for every edge in supports.

        Channel gating: edges with empty U AND empty V are skipped
        since LI-CTE = 0 by definition when there is no interaction
        support on this channel this turn.

        Only edges that are also in E0 are accepted — active edges
        outside the feasible scaffold are ignored.

        Call once per channel per turn BEFORE compute_all.
        """
        for (src, tgt), support in supports.items():
            # Channel gating
            if not support.U and not support.V:
                continue

            # Feasibility gate — must be in E0
            if self._feasible and (src, tgt) not in self._feasible:
                continue

            U = encode_payload(support.U, self._encode_dim)
            V = encode_payload(support.V, self._encode_dim)
            H = encode_payload(support.H, self._encode_dim)

            key = (src, tgt, channel)
            self._buffer[key].append((turn, U, V, H))

            # Bound buffer at L+1 — O(1) maintenance
            if len(self._buffer[key]) > self._L + 1:
                self._buffer[key].pop(0)

            self._active.add(key)

    def compute_all(
        self,
        feasible_edges : list[tuple[str, str]],
        turn           : int,
    ) -> dict[tuple[str, str, Channel], float]:
        """
        Compute LI-CTE_ij^(c)(t) for all active edges this turn.

        feasible_edges is the full E0 edge list from StructuralGraph.
        We intersect it with _active — only edges that had interaction
        support pushed AND are in E0 are computed. Everything else is 0
        by definition.

        Returns dict[(src, tgt, channel)] -> float >= 0.
        """
        # Intersection: active this turn AND in E0
        feasible_set = set(feasible_edges)
        to_compute   = {
            key for key in self._active
            if (key[0], key[1]) in feasible_set
        }

        result: dict[tuple[str, str, Channel], float] = {}
        for key in to_compute:
            src, tgt, channel = key
            result[key] = self._compute_one(src, tgt, channel, turn)

        # Clear active set for next turn
        self._active.clear()

        return result

    # ── Core computation ───────────────────────────────────────────────────

    def _compute_one(
        self,
        src     : str,
        tgt     : str,
        channel : Channel,
        turn    : int,
    ) -> float:
        """
        LI-CTE_ij^(c)(t) = max_{τ=0}^{L} CTE(t, τ)

        CTE(t, τ) = LRTS( U(t-τ), V(t), H(t-1) )

        V — always current turn.
        U — lagged by τ.
        H — always t-1, not τ-dependent (spec: H_j^(c)(t-1)).
        """
        key = (src, tgt, channel)
        buf = self._buffer[key]
        if not buf:
            return 0.0

        turn_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            t: (U, V, H) for t, U, V, H in buf
        }

        if turn not in turn_map:
            return 0.0

        _, V_current, _ = turn_map[turn]

        # H(t-1): always one step back, not τ-dependent
        prev   = turn - 1
        H_cond = turn_map[prev][2] if prev in turn_map else (
            np.zeros(self._encode_dim, dtype=np.float32)
        )

        best = 0.0

        for tau in range(self._L + 1):
            lagged = turn - tau
            if lagged not in turn_map:
                continue

            U_lagged = turn_map[lagged][0]

            if self._exact_cmi and self._estimator is not None:
                score = self._exact_score(turn_map, turn, tau)
            else:
                score = _lrts(U_lagged, V_current, H_cond, self._eps)

            if score > best:
                best = score

            # Early lag pruning
            if best >= self._EARLY_STOP:
                break

        return float(best)

    def _exact_score(
        self,
        turn_map : dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        turn     : int,
        tau      : int,
    ) -> float:
        """
        Exact CMI estimation (ablation/validation mode only).

        Per-sample conditioning: each sample (t, t-tau) uses its own
        H(t-1) as the conditioning term — correctly aligned with the
        formal definition CTE(t, τ) = I(U(t-τ); V(t) | H(t-1)).

        Falls back to LRTS if fewer than 3 samples available.
        """
        dim = self._encode_dim
        U_rows, V_rows, H_rows = [], [], []

        for t, (U, V, _) in turn_map.items():
            lag_t = t - tau
            if lag_t not in turn_map:
                continue
            U_lag = turn_map[lag_t][0]

            # Per-sample H: use H(t-1) for each sample's own t
            # This aligns with the formal definition
            t_prev = t - 1
            H_sample = turn_map[t_prev][2] if t_prev in turn_map else (
                np.zeros(dim, dtype=np.float32)
            )

            U_rows.append(U_lag)
            V_rows.append(V)
            H_rows.append(H_sample)

        if len(U_rows) < 3:
            # Not enough samples — fall back to LRTS at τ=0 of this key
            if turn in turn_map and (turn - tau) in turn_map:
                _, V_cur, _ = turn_map[turn]
                U_lag, _, _ = turn_map[turn - tau]
                t_prev = turn - 1
                H_cond = turn_map[t_prev][2] if t_prev in turn_map else (
                    np.zeros(dim, dtype=np.float32)
                )
                return _lrts(U_lag, V_cur, H_cond, 1e-8)
            return 0.0

        try:
            return float(self._estimator.estimate(
                np.array(U_rows, dtype=np.float32),
                np.array(V_rows, dtype=np.float32),
                np.array(H_rows, dtype=np.float32),
            ))
        except Exception:
            # Estimator failed — fall back to LRTS
            if turn in turn_map and (turn - tau) in turn_map:
                _, V_cur, _ = turn_map[turn]
                U_lag, _, _ = turn_map[turn - tau]
                t_prev = turn - 1
                H_cond = turn_map[t_prev][2] if t_prev in turn_map else (
                    np.zeros(dim, dtype=np.float32)
                )
                return _lrts(U_lag, V_cur, H_cond, 1e-8)
            return 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state. Call between benchmark scenarios."""
        self._buffer.clear()
        self._active.clear()


# ── Factory ────────────────────────────────────────────────────────────────

def build_li_cte_engine(
    config   : DetectorConfig,
    n_agents : int,
) -> LICTEEngine:
    """
    Resolve scale tier and return a ready LICTEEngine.
    Call config.resolve(n_agents) before this.
    """
    config.resolve(n_agents)
    return LICTEEngine(config, n_agents)
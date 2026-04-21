"""
caspian/core/attribution.py

Steps 20-23 — Cumulative cascade graph and attribution inference.

Step 20  Cumulative cascade graph over interval [t0, t1]:
             A_cum = Σ_{τ=t0}^{t1} A_τ
             χ_cum(i,j) = argmax_c  Σ_{τ=t0}^{t1} A_τ^(c)[i,j]
         This is the graph used for all attribution and spine inference.

Step 21  Origin inference:
             O_i = Σ_{τ=t0}^{t0+δ} Σ_j A_τ[i,j]
             Origin = argmax_i O_i
         δ is a short attribution horizon — not a model parameter.
         Only the first δ turns after onset are used because the origin
         acts earliest and its signal is strongest near t0.

Step 22  Bridge inference:
             B_j = Σ_{i,k} A_cum[i,j] · A_cum[j,k] · 1[χ_cum(i,j) ≠ χ_cum(j,k)]
             Bridge = argmax_j B_j
         Strong in-flow + strong out-flow + channel transition at j.
         Returns None if no node has non-zero cross-channel relay flow.

Step 23  Amplifier inference:
             M_i = Σ_j A_cum[i,j]        (cumulative row sum = outflow)
             Amplifier = argmax_i M_i
         Top-k amplifiers retained by M_i.
"""

from __future__ import annotations

import numpy as np

from utils.types import Channel
from core.channel_matrix import ChannelMatrices


class AttributionEngine:
    """
    Accumulates per-turn fused matrices and channel matrices over the
    cascade window [t0, t1], then infers Origin, Bridge, and Amplifier.

    Usage
    -----
        engine = AttributionEngine(agent_ids)

        # During cascade window — call each turn:
        engine.accumulate(cm, turn)

        # After window closes:
        origin     = engine.infer_origin(t0, delta)
        bridge     = engine.infer_bridge()
        amplifiers = engine.infer_amplifiers(top_k=3)
    """

    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = agent_ids
        self._idx      = {a: i for i, a in enumerate(agent_ids)}
        N              = len(agent_ids)

        # Step 20: A_cum = Σ A_τ  over [t0, t1]
        self.A_cum = np.zeros((N, N), dtype=np.float64)

        # Step 20: per-channel cumulative sum for χ_cum
        # A_cum_c[c][i,j] = Σ_{τ=t0}^{t1} A_τ^(c)[i,j]
        self._A_cum_c: dict[Channel, np.ndarray] = {
            c: np.zeros((N, N), dtype=np.float64) for c in Channel
        }

        # Early-window accumulator for origin scoring (step 21)
        # Stores per-turn row sums for turns [t0, t0+δ]
        # key: turn -> per-agent outflow vector
        self._early_outflow: dict[int, np.ndarray] = {}

    @property
    def agent_idx(self) -> dict[str, int]:
        """Public read-only view of the agent -> matrix-index mapping."""
        return self._idx

    # ── Accumulation ───────────────────────────────────────────────────────

    def accumulate(self, cm: ChannelMatrices, turn: int) -> None:
        """
        Add one turn's matrices to A_cum and A_cum_c.
        Record per-agent outflow for early-window origin scoring.

        Call once per turn inside the cascade window [t0, t1].
        """
        # Step 20: A_cum += A_τ
        self.A_cum += cm.A_t.astype(np.float64)

        # Step 20: per-channel accumulation for χ_cum
        for c in Channel:
            self._A_cum_c[c] += cm.A_norm[c].astype(np.float64)

        # Step 21: store per-agent outflow at this turn for δ-window scoring
        # outflow_i(τ) = Σ_j A_τ[i,j]  (row sum of A_τ)
        self._early_outflow[turn] = cm.A_t.sum(axis=1).astype(np.float64)

    # ── Step 20: cumulative dominant channel ──────────────────────────────

    def chi_cum(self, src: str, tgt: str) -> Channel:
        """
        χ_cum(i,j) = argmax_c Σ_{τ=t0}^{t1} A_τ^(c)[i,j]
        Dominant channel for edge (src, tgt) over the cascade window.
        """
        i = self._idx.get(src)
        j = self._idx.get(tgt)
        if i is None or j is None:
            return Channel.COMM
        vals = [float(self._A_cum_c[c][i, j]) for c in Channel]
        return list(Channel)[int(np.argmax(vals))]

    # ── Step 21: Origin inference ──────────────────────────────────────────

    def infer_origin(self, t0: int, delta: int) -> str:
        """
        O_i = Σ_{τ=t0}^{t0+δ} Σ_j A_τ[i,j]
        Origin = argmax_i O_i

        Only turns in the early window [t0, t0+δ] contribute.
        δ is a short attribution horizon — not a model parameter.
        The origin acts earliest so its outbound signal peaks near t0.

        Falls back to full A_cum row sums if early-window data is sparse.
        """
        N             = len(self.agent_ids)
        origin_scores = np.zeros(N, dtype=np.float64)

        # Sum per-agent outflow over [t0, t0+delta]
        found_early = False
        for turn, outflow in self._early_outflow.items():
            if t0 <= turn <= t0 + delta:
                origin_scores += outflow
                found_early = True

        if not found_early:
            # Fallback: use full cumulative row sums
            origin_scores = self.A_cum.sum(axis=1)

        if origin_scores.max() == 0:
            return self.agent_ids[0]

        return self.agent_ids[int(np.argmax(origin_scores))]

    # ── Step 22: Bridge inference ──────────────────────────────────────────

    def infer_bridge(self) -> str | None:
        """
        B_j = Σ_{i,k} A_cum[i,j] · A_cum[j,k] · 1[χ_cum(i,j) ≠ χ_cum(j,k)]
        Bridge = argmax_j B_j

        Identifies the strongest cross-channel pivot node:
        strong incoming flow × strong outgoing flow × channel transition.

        Uses adjacency-pivot structure to avoid O(N³) enumeration:
            For each pivot j:
                B_j = Σ_i A_cum[i,j] · Σ_k A_cum[j,k] · cross_channel_flag(i,j,k)

        Returns None if no node has positive cross-channel relay flow.
        """
        ids  = self.agent_ids
        idx  = self._idx
        N    = len(ids)
        scores = np.zeros(N, dtype=np.float64)

        for j, mid in enumerate(ids):
            score = 0.0
            # All sources feeding into mid
            for i, src in enumerate(ids):
                in_flow = self.A_cum[i, j]
                if in_flow == 0.0:
                    continue
                c_in = self.chi_cum(src, mid)

                # All targets mid feeds into
                for k, tgt in enumerate(ids):
                    out_flow = self.A_cum[j, k]
                    if out_flow == 0.0:
                        continue
                    c_out = self.chi_cum(mid, tgt)

                    # Cross-channel: channel changes at pivot j
                    if c_in != c_out:
                        score += in_flow * out_flow

            scores[j] = score

        if scores.max() == 0.0:
            return None

        return ids[int(np.argmax(scores))]

    # ── Step 23: Amplifier inference ──────────────────────────────────────

    def infer_amplifiers(self, top_k: int = 3) -> list[str]:
        """
        M_i = Σ_j A_cum[i,j]    (cumulative outflow = row sum of A_cum)
        Amplifier = argmax_i M_i

        Returns the top_k agents by cumulative outflow, ranked descending.
        The spec says "or retain top few amplifiers by M_i" — we return
        top_k for flexibility; caller decides how many to use.
        """
        # Row sums of A_cum
        outflows = self.A_cum.sum(axis=1)   # shape (N,)

        if outflows.max() == 0.0:
            return []

        # Argsort descending
        ranked = np.argsort(outflows)[::-1]
        return [self.agent_ids[int(i)] for i in ranked[:top_k]]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all accumulated state.
        Call before starting a new cascade window accumulation.
        """
        N = len(self.agent_ids)
        self.A_cum[:] = 0.0
        for c in Channel:
            self._A_cum_c[c][:] = 0.0
        self._early_outflow.clear()
"""
caspian/core/channel_matrix.py

Steps 8-11 — Channel matrices, normalization, channel-embedded edge
vectors, and the fused influence matrix A_t.

Step 8:  Build four N×N matrices from LI-CTE values
             A_t^(c)[i,j] = LI-CTE_ij^(c)(t)   for (i,j) in E0
             A_t^(c)[i,j] = 0                   otherwise

Step 9:  Normalize each channel matrix per turn
             A_t^(c)[i,j] = A_t^(c)[i,j] / (max_{u,v} A_t^(c)[u,v] + ε)
         ε is machine tolerance only — not a hyperparameter.

Step 10: Channel-embedded edge vector (fixed orthogonal basis)
             e_ij(t) = A_t^(comm)[i,j] * g_comm
                     + A_t^(mem)[i,j]  * g_mem
                     + A_t^(tool)[i,j] * g_tool
                     + A_t^(exec)[i,j] * g_exec
         g_comm=[1,0,0,0], g_mem=[0,1,0,0], g_tool=[0,0,1,0], g_exec=[0,0,0,1]
         These are fixed, not learned. Non-interfering by construction.

Step 11: Fused scalar influence matrix
             A_t[i,j] = ||e_ij(t)||_1
         Since all components are nonneg and basis vectors are orthogonal,
         this equals the sum of the four normalized channel values.

         Dominant channel label:
             χ_t[i,j] = argmax_c A_t^(c)[i,j]

Output of this file feeds:
    core/metrics.py  — uses A_t for λ1, r, p
    core/fusion.py   — uses χ_t for cross-channel propagation p(t)
    core/attribution.py — accumulates A_t into A_cum
"""

from __future__ import annotations

import numpy as np

from utils.types import Channel
from core.graph import StructuralGraph


# Fixed orthogonal channel basis vectors (step 4, not learned)
CHANNEL_BASIS: dict[Channel, np.ndarray] = {
    Channel.COMM: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    Channel.MEM:  np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    Channel.TOOL: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    Channel.EXEC: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


class ChannelMatrices:
    """
    Container for one turn's fully processed channel data.

    Attributes
    ----------
    A_raw  : dict[Channel, ndarray(N,N)]  — raw LI-CTE values (step 8)
    A_norm : dict[Channel, ndarray(N,N)]  — per-channel normalized (step 9)
    A_t    : ndarray(N,N)                 — fused scalar matrix (step 11)
    chi_t  : dict[(str,str), Channel]     — dominant channel per edge (step 11)
    """

    def __init__(self, N: int, agent_ids: list[str]) -> None:
        self.N          = N
        self.agent_ids  = agent_ids
        self._idx       = {a: i for i, a in enumerate(agent_ids)}

        # Step 8: raw matrices — one per channel
        self.A_raw: dict[Channel, np.ndarray] = {
            c: np.zeros((N, N), dtype=np.float32) for c in Channel
        }
        # Step 9: normalized matrices
        self.A_norm: dict[Channel, np.ndarray] = {
            c: np.zeros((N, N), dtype=np.float32) for c in Channel
        }
        # Step 11: fused scalar matrix
        self.A_t: np.ndarray = np.zeros((N, N), dtype=np.float32)

        # Step 11: dominant channel per named edge
        self.chi_t: dict[tuple[str, str], Channel] = {}

    # ── Build from LI-CTE values ───────────────────────────────────────────

    def build(
        self,
        li_cte_values : dict[tuple[str, str, Channel], float],
        graph         : StructuralGraph,
        eps           : float = 1e-8,
    ) -> None:
        """
        Full pipeline: LI-CTE values → steps 8, 9, 10, 11.

        Parameters
        ----------
        li_cte_values : dict[(src, tgt, channel)] -> float
            Output of LICTEEngine.compute_all(). Only active edges
            are present — all other feasible edges implicitly have value 0.
        graph : StructuralGraph
            Provides E0 and agent ordering.
        eps : float
            ε for normalization denominator (config.eps_norm).
        """
        self._fill_raw(li_cte_values)
        self._normalize(eps)
        self._fuse_and_label(graph)

    # ── Step 8: fill raw matrices ──────────────────────────────────────────

    def _fill_raw(
        self,
        li_cte_values: dict[tuple[str, str, Channel], float],
    ) -> None:
        """
        A_t^(c)[i,j] = LI-CTE_ij^(c)(t)  for all (src, tgt, c) in values.
        All other entries remain 0.
        LI-CTE values are clipped to >= 0 (spec: nonnegative matrices).
        """
        # Reset to zero first — matrices reused across turns
        for c in Channel:
            self.A_raw[c][:] = 0.0

        for (src, tgt, channel), value in li_cte_values.items():
            i = self._idx.get(src)
            j = self._idx.get(tgt)
            if i is None or j is None or i == j:
                continue
            # Spec: nonnegative — clip at 0
            self.A_raw[channel][i, j] = max(float(value), 0.0)

    # ── Step 9: per-channel normalization ─────────────────────────────────

    def _normalize(self, eps: float) -> None:
        """
        A_t^(c)[i,j] = A_t^(c)[i,j] / (max_{u,v} A_t^(c)[u,v] + ε)

        Spec: ε is numerical stability only, not a hyperparameter.
        Each channel is normalized independently — magnitudes differ
        across channels and normalizing makes them comparable.
        """
        for c in Channel:
            raw      = self.A_raw[c]
            max_val  = float(raw.max())
            self.A_norm[c] = raw / (max_val + eps)

    # ── Steps 10-11: embed, fuse, label ───────────────────────────────────

    def _fuse_and_label(self, graph: StructuralGraph) -> None:
        """
        Step 10: e_ij(t) = Σ_c A_norm^(c)[i,j] * g_c
                           (4-dimensional channel-embedded vector per edge)

        Step 11: A_t[i,j]  = ||e_ij(t)||_1
                            = sum of four normalized channel values
                              (equivalent because basis is orthogonal and
                               all components are nonneg)

                 χ_t[i,j] = argmax_c A_norm^(c)[i,j]
        """
        # Reset
        self.A_t[:] = 0.0
        self.chi_t.clear()

        for i, src in enumerate(self.agent_ids):
            for j, tgt in enumerate(self.agent_ids):
                if i == j:
                    continue

                # Collect per-channel normalized values for this edge
                channel_vals = np.array(
                    [self.A_norm[c][i, j] for c in Channel],
                    dtype=np.float32,
                )  # shape (4,)

                # Step 11: fused scalar = L1 norm of channel-embedded vector
                # Since basis is orthogonal and values nonneg:
                # ||e_ij||_1 = Σ_c A_norm^(c)[i,j] * ||g_c||_1
                #            = Σ_c A_norm^(c)[i,j]   (each g_c has ||.||_1 = 1)
                fused = float(channel_vals.sum())
                self.A_t[i, j] = fused

                # Step 11: dominant channel
                if fused > 0.0:
                    dom_idx = int(np.argmax(channel_vals))
                    self.chi_t[(src, tgt)] = list(Channel)[dom_idx]

    # ── Accessors ──────────────────────────────────────────────────────────

    def dominant_channel(self, src: str, tgt: str) -> Channel:
        """
        χ_t[i,j] — dominant channel for edge (src, tgt) this turn.
        Returns Channel.COMM as default if edge has zero influence.
        """
        return self.chi_t.get((src, tgt), Channel.COMM)

    def active_edges(self, eps_active: float) -> list[tuple[str, str]]:
        """
        E_t^act = { (i,j) : A_t[i,j] > ε_A }
        Used by metrics.py for cross-channel propagation metric p(t).
        """
        rows, cols = np.where(self.A_t > eps_active)
        return [
            (self.agent_ids[i], self.agent_ids[j])
            for i, j in zip(rows.tolist(), cols.tolist())
        ]

    def channel_mass(self, channel: Channel) -> float:
        """
        M_c(t) = Σ_{i,j} A_t^(c)[i,j]  (step 17)
        Total influence mass on channel c this turn.
        Uses normalized matrix so all channels are comparable.
        """
        return float(self.A_norm[channel].sum())


# ── Factory ────────────────────────────────────────────────────────────────

def build_channel_matrices(
    graph         : StructuralGraph,
    li_cte_values : dict[tuple[str, str, Channel], float],
    eps           : float = 1e-8,
) -> ChannelMatrices:
    """
    Convenience factory: allocate ChannelMatrices and run full pipeline.

    Parameters
    ----------
    graph         : StructuralGraph — provides N and agent ordering
    li_cte_values : output of LICTEEngine.compute_all()
    eps           : normalization tolerance (config.eps_norm)

    Returns
    -------
    ChannelMatrices with A_raw, A_norm, A_t, chi_t all populated.
    """
    cm = ChannelMatrices(N=graph.N, agent_ids=graph.agent_ids)
    cm.build(li_cte_values, graph, eps)
    return cm
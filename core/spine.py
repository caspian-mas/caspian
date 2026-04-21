"""
caspian/core/spine.py

Step 24 — Top-K cascade propagation spine inference.

Spec step 24:
    Treat A_cum as a weighted directed graph.
    For a simple path P = (v0 → v1 → ... → vℓ):

        Score(P) = Σ_{m=0}^{ℓ-1} log(A_cum[v_m, v_{m+1}] + ε)

    This is the log-sum of edge weights — equivalent to maximising
    the product of edge weights but numerically stable.

    Generate candidate simple paths:
        - starting from Origin
        - optionally passing through Bridge
        - bounded by maximum path length Lp

    Rank by Score(P), return top K.

    Note on Lp (spec explicit):
        Do not treat Lp as random. It is a computational truncation choice.
        Logical bound: Lp ≤ min(cascade_interval_length, graph_diameter).

Three path generation strategies, selected by scale tier:
    SMALL  — full simple-path enumeration  (exact, capped at max_paths)
    MEDIUM — beam search                   (greedy, width=beam_width)
    LARGE  — random-walk sampling          (n_samples walks)

All strategies feed into the same scoring and ranking step.
"""

from __future__ import annotations

import numpy as np

from utils.types import Spine
from utils.config import (
    DetectorConfig,
    SmallScaleConfig,
    MediumScaleConfig,
    LargeScaleConfig,
)
from utils.graph_utils import (
    graph_diameter,
    enumerate_simple_paths,
    beam_search_paths,
    sampled_paths,
)
from core.attribution import AttributionEngine


class SpineInference:
    """
    Infers Top-K cascade propagation spines from A_cum.

    Parameters
    ----------
    config : DetectorConfig — K, Lp, eps_norm, scale tier all from here
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    def infer(
        self,
        attribution  : AttributionEngine,
        origin       : str,
        bridge       : str | None,
        t0           : int,
        t1           : int,
    ) -> list[Spine]:
        """
        Generate and rank candidate propagation paths in A_cum.

        Parameters
        ----------
        attribution : AttributionEngine with fully accumulated A_cum
        origin      : detected origin agent (start of all paths)
        bridge      : detected bridge agent (must-pass filter, or None)
        t0, t1      : cascade window bounds — used to compute Lp cap

        Returns
        -------
        list[Spine] ranked by Score(P) descending, length <= K
        """
        A_cum     = attribution.A_cum
        agent_ids = attribution.agent_ids
        K         = self._config.K
        eps       = self._config.eps_norm

        # ── Lp: spec says min(cascade_length, graph_diameter) ─────────────
        cascade_len = max(t1 - t0 + 1, 1)  # inclusive: [t0, t1] has t1-t0+1 turns
        diameter    = graph_diameter(A_cum, agent_ids)
        Lp_bound    = min(cascade_len, diameter)

        # Config Lp may further restrict; None means use the bound
        if self._config.Lp is not None:
            Lp = min(self._config.Lp, Lp_bound)
        else:
            Lp = Lp_bound

        # Lp is measured in EDGES (not nodes).
        # A path v0→v1→...→vℓ has ℓ edges and ℓ+1 nodes.
        # All candidate generators in graph_utils.py use cutoff=Lp as edge count.
        Lp = max(Lp, 1)   # at least one edge

        # ── Generate candidate paths by scale tier ─────────────────────────
        sc = self._config.active_scale_config

        if isinstance(sc, SmallScaleConfig):
            paths = enumerate_simple_paths(
                A          = A_cum,
                agent_ids  = agent_ids,
                origin     = origin,
                Lp         = Lp,
                bridge     = bridge,
                max_paths  = sc.max_spine_paths,
            )
        elif isinstance(sc, MediumScaleConfig):
            paths = beam_search_paths(
                A          = A_cum,
                agent_ids  = agent_ids,
                origin     = origin,
                Lp         = Lp,
                beam_width = sc.beam_width,
                bridge     = bridge,
            )
        else:  # LargeScaleConfig
            paths = sampled_paths(
                A          = A_cum,
                agent_ids  = agent_ids,
                origin     = origin,
                Lp         = Lp,
                n_samples  = sc.spine_samples,
                bridge     = bridge,
            )

        # ── Score all candidate paths ──────────────────────────────────────
        idx    = attribution.agent_idx  # public property
        scored = self._score_paths(paths, A_cum, idx, eps)

        # ── Rank, deduplicate, return top K ───────────────────────────────
        return self._top_k(scored, K)

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_paths(
        self,
        paths     : list[list[str]],
        A_cum     : np.ndarray,
        idx       : dict[str, int],
        eps       : float,
    ) -> list[Spine]:
        """
        Score(P) = Σ_{m=0}^{ℓ-1} log(A_cum[v_m, v_{m+1}] + ε)

        Spec: "equivalent to maximising the product of edge weights,
               but numerically stable."

        Paths with fewer than 2 nodes are skipped.
        Paths with any unknown agent are skipped.
        """
        scored: list[Spine] = []

        for path in paths:
            if len(path) < 2:
                continue

            score = 0.0
            valid = True

            for m in range(len(path) - 1):
                i = idx.get(path[m])
                j = idx.get(path[m + 1])
                if i is None or j is None:
                    valid = False
                    break
                score += np.log(float(A_cum[i, j]) + eps)

            if valid:
                scored.append(Spine(nodes=list(path), score=float(score)))

        return scored

    # ── Top-K deduplication ────────────────────────────────────────────────

    def _top_k(self, scored: list[Spine], K: int) -> list[Spine]:
        """
        Sort by score descending, deduplicate by exact node sequence,
        return top K.

        Deduplication: beam search and sampling may produce the same
        path multiple times — we keep the first (highest-scored) occurrence.
        """
        scored.sort(key=lambda s: s.score, reverse=True)

        seen    : set[tuple[str, ...]] = set()
        top_k   : list[Spine]          = []

        for spine in scored:
            key = tuple(spine.nodes)
            if key not in seen:
                seen.add(key)
                top_k.append(spine)
            if len(top_k) >= K:
                break

        return top_k
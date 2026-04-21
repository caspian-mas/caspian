"""
caspian/core/detector.py

CaspianDetector — the single public entry point that orchestrates the
full algorithm turn by turn.

Compact algorithm (spec step 27):
    1.  Build G0 = (V, E0)
    2.  For each turn t, each feasible edge i→j: extract channel states
    3.  Compute LI-CTE_ij^(c)(t) for c ∈ {comm, mem, tool, exec}
    4.  Normalize channel matrices, build channel-embedded edge vectors
    5.  Fuse to scalar A_t and dominant labels χ_t
    6.  Compute λ1(t), r(t), p(t)
    7.  Self-normalize over last W turns using rolling median and MAD
    8.  Z(t) = min(z̃_λ, z̃_r, z̃_p)
    9.  If Z(t) > τ, declare cascade onset
    10. Use windowed slopes and channel timing to classify type
    11. Aggregate A_t over cascade window into A_cum
    12. Infer Origin, Bridge, Amplifier
    13. Rank Top-K simple paths in A_cum by log-flow score

Final output per cascade:
    (t0, t1, type, Origin, Bridge, Amplifier(s), Top-K Spines)

Usage
-----
    from caspian.core.detector import CaspianDetector
    from caspian.utils.config import default_config
    from caspian.utils.types import Topology

    detector = CaspianDetector(
        agent_ids = ["agent_0", "agent_1", "agent_2"],
        config    = default_config(Topology.MESH),
    )

    for turn, events in event_stream:
        result = detector.step(turn, events)
        if result is not None:
            print(result)   # CascadeResult
"""

from __future__ import annotations

from utils.types import (
    Channel,
    CascadeResult,
    CascadeType,
    InteractionEvent,
    Topology,
)
from utils.config import DetectorConfig, default_config
from utils.logging import (
    configure_logging,
    log_onset,
    log_cascade_window,
    log_attribution,
    log_spines,
    log_turn_metrics,
)

from core.graph import StructuralGraph, topology_factory
from core.channel_states import ChannelHistory, extract_late_interaction_supports
from core.li_cte import LICTEEngine, build_li_cte_engine
from core.channel_matrix import ChannelMatrices, build_channel_matrices
from core.metrics import MetricsEngine
from core.onset import OnsetDetector, CascadeWindow
from core.attribution import AttributionEngine
from core.spine import SpineInference


class CaspianDetector:
    """
    Stateful turn-by-turn cascade detector.

    Parameters
    ----------
    agent_ids  : ordered list of agent identifier strings — defines V
    config     : DetectorConfig — all monitor choices in one object.
                 If None, default_config() is used.
    graph      : pre-built StructuralGraph. If None, built from
                 config.topology + adjacency/custom_E0/subgraphs.
    adjacency  : parent->children map for HIERARCHY/STAR/PIPELINE
    custom_E0  : raw N×N binary array for CUSTOM topology
    subgraphs  : list of sub-topology triples for HYBRID topology
    log_level  : "DEBUG" | "INFO" | "WARNING"
    """

    def __init__(
        self,
        agent_ids  : list[str],
        config     : DetectorConfig | None = None,
        graph      : StructuralGraph | None = None,
        adjacency  : dict | None = None,
        custom_E0  = None,
        subgraphs  = None,
        log_level  : str = "INFO",
    ) -> None:
        configure_logging(log_level)

        # Config — resolve scale tier from N
        self._config = config or default_config()
        self._config.resolve(len(agent_ids))

        # Step 1: build G0
        if graph is not None:
            self._graph = graph
        else:
            self._graph = topology_factory(
                agent_ids = agent_ids,
                topology  = self._config.topology,
                adjacency = adjacency,
                custom_E0 = custom_E0,
                subgraphs = subgraphs,
            )

        # Step 2: channel history (H_j^(c)(t-1) conditioning term)
        self._history = ChannelHistory(max_len=self._config.L + 2)

        # Step 3: LI-CTE engine
        self._li_cte = build_li_cte_engine(self._config, len(agent_ids))
        self._li_cte.set_feasible_edges(self._graph.feasible_edges_named())

        # Steps 4-5, 10-11: channel matrix + fusion (built fresh each turn, no state)
        # build_channel_matrices() runs the full pipeline:
        #   step 8:  A_t^(c)[i,j] = LI-CTE_ij^(c)(t)
        #   step 9:  normalize per channel
        #   step 10: e_ij(t) = Σ_c A_norm^(c)[i,j] * g_c
        #   step 11: A_t[i,j] = ||e_ij(t)||_1,  χ_t[i,j] = argmax_c

        # Steps 6-7: metrics engine (holds rolling windows)
        self._metrics = MetricsEngine(self._config)

        # Steps 8-9: onset detector
        self._onset = OnsetDetector(self._config, self._metrics)

        # Steps 11-12: attribution engine (accumulates A_cum)
        self._attribution = AttributionEngine(agent_ids)

        # Step 13: spine inference (stateless)
        self._spine = SpineInference(self._config)

        # Internal state
        self._in_cascade     = False
        self._t0             = 0
        # Per-turn TurnMetrics accumulated during active cascade window
        # Attached to CascadeResult on emit — needed for analysis + figures
        self._window_metrics : list = []

    # ── Main per-turn step ─────────────────────────────────────────────────

    def step(
        self,
        turn   : int,
        events : list[InteractionEvent],
    ) -> CascadeResult | None:
        """
        Process one turn. Returns a CascadeResult when a cascade window
        closes, otherwise None.

        Parameters
        ----------
        turn   : current turn index (monotonically increasing)
        events : all InteractionEvents for this turn (one per active agent)

        Returns
        -------
        CascadeResult if a cascade window just closed, else None.
        """
        # Index events by agent_id for O(1) lookup
        events_by_agent = {e.agent_id: e for e in events}

        # ── Steps 2-3: extract supports and compute LI-CTE ────────────────
        li_cte_values: dict = {}
        for channel in Channel:
            # Step 2: extract (U, V, H) triples for all active edges
            supports = extract_late_interaction_supports(
                events  = events_by_agent,
                history = self._history,
                graph   = self._graph,
                channel = channel,
                turn    = turn,
            )
            # Step 3: push into LI-CTE engine (encodes and buffers)
            self._li_cte.push_supports(supports, channel, turn)

        # Compute LI-CTE for all active edges × 4 channels
        li_cte_values = self._li_cte.compute_all(
            self._graph.feasible_edges_named(), turn
        )

        # ── Steps 8-11: channel matrices, normalization, embedding, fusion ─
        cm = build_channel_matrices(
            graph         = self._graph,
            li_cte_values = li_cte_values,
            eps           = self._config.eps_norm,
        )

        # ── Steps 6-7: compute metrics and self-normalize ─────────────────
        turn_metrics = self._metrics.compute(cm, turn)

        log_turn_metrics(
            turn     = turn,
            lambda1  = turn_metrics.lambda1,
            r        = turn_metrics.r,
            p        = turn_metrics.p,
            z_lambda = turn_metrics.z_lambda,
            z_r      = turn_metrics.z_r,
            z_p      = turn_metrics.z_p,
            Z        = turn_metrics.Z,
        )

        # ── Steps 8-9: onset detection ────────────────────────────────────
        onset_fired, closed_window = self._onset.update(turn_metrics)

        if onset_fired and not self._in_cascade:
            self._in_cascade = True
            self._t0         = turn
            self._attribution.reset()
            log_onset(turn, turn_metrics.Z, self._config.tau)

        # ── Step 11: accumulate A_t into A_cum while in cascade ───────────
        if self._in_cascade:
            self._attribution.accumulate(cm, turn)
            self._window_metrics.append(turn_metrics)

        # ── Step 10 + 12-13: emit result when window closes ───────────────
        result: CascadeResult | None = None

        if closed_window is not None and self._in_cascade:
            result = self._emit_result(closed_window)
            self._in_cascade = False

        # ── Update channel history AFTER computing supports ────────────────
        # H_j^(c)(t-1) must reflect state up to t-1, not t.
        # Push current turn's events into history at end of turn.
        for event in events:
            self._history.push(event)

        return result

    # ── Result emission ────────────────────────────────────────────────────

    def _emit_result(self, window: CascadeWindow) -> CascadeResult:
        """
        Steps 10, 12, 13 — classify, attribute, infer spines, emit.
        Called exactly once per closed cascade window.
        """
        t0 = window.t0
        t1 = window.t1

        # Step 10: classify single-step vs multi-step
        cascade_type = self._onset.classify(window)

        # Step 12: infer Origin, Bridge, Amplifier
        origin     = self._attribution.infer_origin(t0, self._config.delta)
        bridge     = self._attribution.infer_bridge()
        amplifiers = self._attribution.infer_amplifiers(top_k=3)

        log_attribution(origin, bridge, amplifiers)
        log_cascade_window(t0, t1, cascade_type.value)

        # Step 13: rank Top-K spines
        spines = self._spine.infer(
            attribution = self._attribution,
            origin      = origin,
            bridge      = bridge,
            t0          = t0,
            t1          = t1,
        )

        log_spines(spines)

        # Channel mass histories for result record
        channel_mass = {
            c.value: self._metrics.channel_mass_history(c)
            for c in Channel
        }

        # Capture and clear the per-turn metric trace for this window
        window_metrics       = list(self._window_metrics)
        self._window_metrics.clear()

        return CascadeResult(
            t0           = t0,
            t1           = t1,
            cascade_type = cascade_type,
            origin       = origin,
            bridge       = bridge,
            amplifiers   = amplifiers,
            spines       = spines,
            turn_metrics = window_metrics,
            channel_mass = channel_mass,
        )

    # ── Convenience ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset all stateful components to initial state.
        Call between benchmark scenarios — cheaper than constructing
        a new detector.
        """
        self._history.reset()
        self._li_cte.reset()
        self._metrics.reset()
        self._onset.reset()
        self._attribution.reset()
        self._in_cascade = False
        self._t0         = 0

    @property
    def graph(self) -> StructuralGraph:
        return self._graph

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @property
    def in_cascade(self) -> bool:
        return self._in_cascade
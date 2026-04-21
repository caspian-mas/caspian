"""
caspian/core/graph.py

Step 2 — Initial decentralized MAS graph G0 = (V, E0).

Spec definition:
    V = [0, 1, ..., N-1]
    E0[i,j] = 1  if agent i can in principle influence agent j
               through at least one of the four channels.
    E0[i,j] = 0  otherwise.

An edge i→j is included if ANY of the following hold:
    1. i can message or expose content to j          (comm)
    2. i's memory writes can later be retrieved by j (mem)
    3. i's tool output can be consumed by j          (tool)
    4. i's execution outcome can alter j's action    (exec)

G0 is NOT the attack graph. It is only the feasible influence scaffold.
All per-turn computations (LI-CTE, channel matrices, metrics) operate
only on edges in E0.

topology_factory builds E0 for the five named topologies plus custom.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from utils.types import Topology


@dataclass
class StructuralGraph:
    """
    G0 = (V, E0).

    Attributes
    ----------
    agent_ids : list[str]
        Ordered agent identifiers. Index i in this list == row/col i in E0.
    E0 : np.ndarray  shape (N, N)  dtype uint8
        Binary feasibility matrix. E0[i,j] = 1 iff i can influence j.
        Diagonal is always 0 (no self-loops).
    """

    agent_ids : list[str]
    E0        : np.ndarray

    # ── derived properties ────────────────────────────────────────────────

    @property
    def N(self) -> int:
        return len(self.agent_ids)

    @property
    def idx(self) -> dict[str, int]:
        """agent_id -> row/col index in E0."""
        return {a: i for i, a in enumerate(self.agent_ids)}

    def feasible_edges(self) -> list[tuple[int, int]]:
        """
        All (i, j) index pairs where E0[i,j] == 1.
        Used by the per-turn loop in core/channel_states.py and li_cte.py.
        """
        rows, cols = np.where(self.E0 == 1)
        return list(zip(rows.tolist(), cols.tolist()))

    def feasible_edges_named(self) -> list[tuple[str, str]]:
        """Same as feasible_edges() but returns (agent_id_i, agent_id_j)."""
        return [
            (self.agent_ids[i], self.agent_ids[j])
            for i, j in self.feasible_edges()
        ]

    def __repr__(self) -> str:
        n_edges = int(self.E0.sum())
        return (
            f"StructuralGraph(N={self.N}, "
            f"edges={n_edges}, "
            f"density={n_edges / max(self.N*(self.N-1), 1):.2f})"
        )


# ── Topology factory (main entry point) ───────────────────────────────────

def topology_factory(
    agent_ids : list[str],
    topology  : Topology,
    *,
    # HIERARCHY / STAR / PIPELINE: explicit parent -> [children] map.
    # If None, a default wiring is inferred (see each builder).
    adjacency : dict[str, list[str]] | None = None,
    # CUSTOM: caller supplies the raw N×N binary matrix directly.
    custom_E0 : np.ndarray | None = None,
    # HYBRID: list of (sub_topology, sub_agent_ids, sub_adjacency) triples
    # whose E0 matrices are OR-ed together into the final E0.
    subgraphs : list[tuple[Topology, list[str], dict | None]] | None = None,
) -> StructuralGraph:
    """
    Build G0 for any supported topology.

    All builders:
      - enforce no self-loops (diagonal zeroed)
      - return a StructuralGraph with uint8 E0

    Parameters
    ----------
    agent_ids : ordered list of all agent ids — defines V and index mapping
    topology  : one of Topology.{MESH, HIERARCHY, PIPELINE, STAR, HYBRID, CUSTOM}
    adjacency : for HIERARCHY / STAR / PIPELINE — parent -> children wiring.
                If omitted, a sensible default is used (see each builder).
    custom_E0 : for CUSTOM — raw (N, N) binary numpy array.
    subgraphs : for HYBRID — list of (topology, agent_subset, adjacency)
                triples. Their E0s are unioned.
    """
    N   = len(agent_ids)
    _idx = {a: i for i, a in enumerate(agent_ids)}

    if topology == Topology.MESH:
        E0 = _build_mesh(N)

    elif topology == Topology.PIPELINE:
        E0 = _build_pipeline(N, agent_ids, adjacency, _idx)

    elif topology == Topology.HIERARCHY:
        E0 = _build_hierarchy(N, agent_ids, adjacency, _idx)

    elif topology == Topology.STAR:
        E0 = _build_star(N, agent_ids, adjacency, _idx)

    elif topology == Topology.HYBRID:
        E0 = _build_hybrid(N, agent_ids, subgraphs or [], _idx)

    elif topology == Topology.CUSTOM:
        if custom_E0 is None:
            raise ValueError("custom_E0 must be provided for Topology.CUSTOM")
        E0 = np.asarray(custom_E0, dtype=np.uint8)
        if E0.shape != (N, N):
            raise ValueError(
                f"custom_E0 shape {E0.shape} does not match N={N}"
            )

    else:
        raise ValueError(f"Unsupported topology: {topology!r}")

    # Spec: G0 has no self-loops — E0[i,i] must be 0
    np.fill_diagonal(E0, 0)

    return StructuralGraph(agent_ids=list(agent_ids), E0=E0)


# ── Individual topology builders ───────────────────────────────────────────

def _build_mesh(N: int) -> np.ndarray:
    """
    MESH — all-to-all directed edges.
    E0[i,j] = 1 for all i ≠ j.
    Diagonal zeroed by topology_factory after return.

    Typical use: AutoGen GroupChat, any fully-connected MAS.
    """
    E0 = np.ones((N, N), dtype=np.uint8)
    return E0


def _build_pipeline(
    N         : int,
    agent_ids : list[str],
    adjacency : dict[str, list[str]] | None,
    idx       : dict[str, int],
) -> np.ndarray:
    """
    PIPELINE — linear chain A0 → A1 → A2 → ... → A_{N-1}.

    If adjacency is provided it defines explicit directed edges,
    allowing non-linear pipelines (branching, merging).
    If None, edges are i → i+1 for i in 0..N-2.

    Typical use: CAMEL role-playing chains, sequential task pipelines.
    """
    E0 = np.zeros((N, N), dtype=np.uint8)
    if adjacency:
        for src, targets in adjacency.items():
            if src not in idx:
                continue
            for tgt in targets:
                if tgt in idx:
                    E0[idx[src], idx[tgt]] = 1
    else:
        for i in range(N - 1):
            E0[i, i + 1] = 1
    return E0


def _build_hierarchy(
    N         : int,
    agent_ids : list[str],
    adjacency : dict[str, list[str]] | None,
    idx       : dict[str, int],
) -> np.ndarray:
    """
    HIERARCHY — tree-shaped edges: parent ↔ children (bidirectional).

    Both directions are included because:
      - parent → child  : task delegation, instruction passing
      - child → parent  : result reporting, status updates
    Both satisfy the spec's edge inclusion criteria.

    If adjacency is None: agent_ids[0] is treated as root,
    all others as direct children.

    Typical use: CrewAI hierarchical process, MetaGPT team structure.
    """
    E0 = np.zeros((N, N), dtype=np.uint8)
    if adjacency:
        for parent, children in adjacency.items():
            if parent not in idx:
                continue
            pi = idx[parent]
            for child in children:
                if child not in idx:
                    continue
                ci = idx[child]
                E0[pi, ci] = 1   # parent → child
                E0[ci, pi] = 1   # child  → parent
    else:
        # Default: star-shaped hierarchy, agent_ids[0] as root
        root = 0
        for i in range(1, N):
            E0[root, i] = 1
            E0[i, root] = 1
    return E0


def _build_star(
    N         : int,
    agent_ids : list[str],
    adjacency : dict[str, list[str]] | None,
    idx       : dict[str, int],
) -> np.ndarray:
    """
    STAR — central hub ↔ all spokes.  Spokes do NOT connect to each other.

    Hub is the first key in adjacency, or agent_ids[0] if adjacency is None.
    Bidirectional hub ↔ spoke edges because the hub both dispatches
    (exec channel) and receives results (comm/tool channels).

    Typical use: LangGraph supervisor, single-orchestrator patterns.
    """
    E0 = np.zeros((N, N), dtype=np.uint8)
    if adjacency:
        hub = list(adjacency.keys())[0]
        if hub not in idx:
            raise ValueError(f"Hub agent '{hub}' not in agent_ids")
        hi = idx[hub]
        for spoke in adjacency[hub]:
            if spoke not in idx:
                continue
            si = idx[spoke]
            E0[hi, si] = 1   # hub → spoke
            E0[si, hi] = 1   # spoke → hub
    else:
        hub = 0
        for i in range(1, N):
            E0[hub, i] = 1
            E0[i, hub] = 1
    return E0


def _build_hybrid(
    N         : int,
    agent_ids : list[str],
    subgraphs : list[tuple[Topology, list[str], dict | None]],
    idx       : dict[str, int],
) -> np.ndarray:
    """
    HYBRID — union of E0 matrices from multiple sub-topologies.

    Each subgraph is a (topology, sub_agent_ids, sub_adjacency) triple.
    The sub-agents must be a subset of agent_ids.
    Edges from all subgraphs are OR-ed into the global E0.

    Typical use: MetaGPT (hierarchy inside team + shared memory mesh
    across teams), any MAS with mixed communication patterns.
    """
    E0 = np.zeros((N, N), dtype=np.uint8)

    for sub_topo, sub_agents, sub_adj in subgraphs:
        # Build G0 for the sub-topology
        sub_graph = topology_factory(
            sub_agents,
            sub_topo,
            adjacency=sub_adj,
        )
        # Map sub-indices back to global indices and OR into E0
        for si, src in enumerate(sub_agents):
            for sj, tgt in enumerate(sub_agents):
                if sub_graph.E0[si, sj] == 1:
                    gi = idx.get(src)
                    gj = idx.get(tgt)
                    if gi is not None and gj is not None:
                        E0[gi, gj] = 1

    return E0
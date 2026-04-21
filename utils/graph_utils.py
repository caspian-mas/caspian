"""
caspian/utils/graph_utils.py

Graph utilities serving three parts of the algorithm:

  Step 12.1/12.2 — eigenvalue computation on A_t
      λ1(t) = λ_max(A_t)           [Perron root]
      λ2(t) = second largest |eigenvalue|
      r(t)  = λ2(t) / (λ1(t) + ε)

  Step 11 / note on Lp — graph diameter for auto-capping Lp
      Lp ≤ min(cascade_interval_length, diameter(G0))

  Step 24 — candidate path generation for spine inference
      Paths start at Origin, optionally pass through Bridge,
      bounded by Lp, scored externally by log-flow score.

Three path strategies, selected by scale tier:
  SMALL  — full simple-path enumeration (networkx all_simple_paths)
  MEDIUM — beam search (greedy, width=beam_width)
  LARGE  — random-walk sampling
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs


# ── Eigenvalue computation (steps 12.1, 12.2) ─────────────────────────────

def top_two_eigenvalues(
    A             : np.ndarray,
    method        : str = "dense",
    sparse_k      : int = 6,
    power_steps   : int = 30,
) -> tuple[float, float]:
    """
    Return (λ1, λ2) — the two largest eigenvalue magnitudes of A.

    Spec step 12.1: λ1(t) = λ_max(A_t)   [Perron root]
    Spec step 12.2: λ2(t) = second largest |eigenvalue|

    Parameters
    ----------
    A           : N×N float ndarray (the fused influence matrix A_t)
    method      : "dense"  — np.linalg.eigvals, exact (SMALL)
                  "sparse" — scipy ARPACK eigs    (MEDIUM)
                  "power"  — power iteration, λ1 only, λ2=0 (LARGE)
    sparse_k    : number of eigenvalues to request from ARPACK
                  (MediumScaleConfig.eigen_k)
    power_steps : iterations for power method
                  (LargeScaleConfig.power_iter_steps)

    Returns
    -------
    (λ1, λ2) both >= 0
    """
    n = A.shape[0]
    if n == 0:
        return 0.0, 0.0

    if method == "dense":
        vals = np.linalg.eigvals(A.astype(np.float64))
        mags = np.sort(np.abs(vals))[::-1]
        λ1 = float(mags[0]) if len(mags) >= 1 else 0.0
        λ2 = float(mags[1]) if len(mags) >= 2 else 0.0
        return λ1, λ2

    elif method == "sparse":
        # ARPACK needs k < n-1; fall back to dense if too small
        k = min(sparse_k, n - 2)
        if k < 2:
            return top_two_eigenvalues(A, method="dense")
        try:
            sp   = csr_matrix(A.astype(np.float64))
            vals = sparse_eigs(sp, k=k, which="LM", return_eigenvectors=False)
            mags = np.sort(np.abs(vals))[::-1]
            λ1   = float(mags[0]) if len(mags) >= 1 else 0.0
            λ2   = float(mags[1]) if len(mags) >= 2 else 0.0
            return λ1, λ2
        except Exception:
            # ARPACK can fail on near-zero matrices; fall back
            return top_two_eigenvalues(A, method="dense")

    elif method == "power":
        λ1 = _power_iteration(A, steps=power_steps)
        # For LARGE scale the spec uses λ2=0 (only λ1 needed for r)
        return λ1, 0.0

    else:
        raise ValueError(f"Unknown eigenvalue method: {method!r}")


def _power_iteration(A: np.ndarray, steps: int = 30) -> float:
    """
    Estimate the spectral radius (largest |eigenvalue|) of A
    via power iteration. O(N^2) per step, no matrix decomposition.
    """
    n   = A.shape[0]
    rng = np.random.default_rng(seed=0)
    v   = rng.random(n)
    v  /= (np.linalg.norm(v) + 1e-12)

    for _ in range(steps):
        v_new = A @ v
        norm  = np.linalg.norm(v_new)
        if norm < 1e-12:
            return 0.0
        v = v_new / norm

    # Rayleigh quotient gives the eigenvalue estimate
    return float(abs(v @ A @ v))


# ── Graph diameter (for Lp auto-cap, step 11 / note on Lp) ────────────────

def graph_diameter(A: np.ndarray, agent_ids: list[str]) -> int:
    """
    Longest shortest directed path in the graph induced by A.

    Spec note on Lp:
        Lp ≤ min(cascade_interval_length, graph_diameter)

    Uses directed reachability — consistent with spine inference which
    follows directed edges in A_cum. Specifically: the diameter is the
    maximum over all (u,v) pairs of the shortest directed path length
    from u to v, considering only reachable pairs.

    Falls back to len(agent_ids) as a safe upper bound when the graph
    is not strongly connected or computation fails.
    """
    G = _matrix_to_nx(A, agent_ids, threshold=0.0)

    try:
        if nx.is_strongly_connected(G):
            # All pairs reachable — true directed diameter
            return nx.diameter(G)
        # Not strongly connected: use longest shortest path
        # over all reachable (u,v) pairs
        max_dist = 0
        for src in G.nodes:
            lengths = nx.single_source_shortest_path_length(G, src)
            if lengths:
                max_dist = max(max_dist, max(lengths.values()))
        return max(max_dist, 1)
    except Exception:
        return len(agent_ids)


# ── Candidate path generation (step 24) ───────────────────────────────────
# All three strategies return list[list[str]] — unscored paths.
# Scoring (Score(P) = Σ log(A_cum[vm,vm+1] + ε)) is done in core/spine.py.

def enumerate_simple_paths(
    A          : np.ndarray,
    agent_ids  : list[str],
    origin     : str,
    Lp         : int,
    bridge     : str | None = None,
    max_paths  : int = 500,
) -> list[list[str]]:
    """
    SMALL scale: enumerate all simple paths in A_cum from `origin`
    with length <= Lp. Optionally filter to paths containing `bridge`.

    Capped at max_paths to prevent combinatorial blow-up even on small graphs.
    """
    G     = _matrix_to_nx(A, agent_ids, threshold=0.0)
    paths : list[list[str]] = []

    for target in agent_ids:
        if target == origin:
            continue
        for path in nx.all_simple_paths(G, origin, target, cutoff=Lp):
            if bridge is not None and bridge not in path:
                continue
            paths.append(list(path))
            if len(paths) >= max_paths:
                return paths

    return paths


def beam_search_paths(
    A          : np.ndarray,
    agent_ids  : list[str],
    origin     : str,
    Lp         : int,
    beam_width : int = 20,
    bridge     : str | None = None,
) -> list[list[str]]:
    """
    MEDIUM scale: greedy beam search over paths from `origin`.
    Each beam state is (cumulative_log_weight, path).
    Expands top beam_width states at each depth level up to Lp.

    Uses log-weights so the score is directly comparable to
    Score(P) in step 24.
    """
    idx = {a: i for i, a in enumerate(agent_ids)}
    eps = 1e-12

    # beam: list of (log_score, path)
    beam      : list[tuple[float, list[str]]] = [(0.0, [origin])]
    completed : list[list[str]]               = []

    for _ in range(Lp):
        candidates: list[tuple[float, list[str]]] = []
        for score, path in beam:
            last = path[-1]
            li   = idx[last]
            for j, nxt in enumerate(agent_ids):
                if nxt in path:          # no cycles
                    continue
                w = float(A[li, j])
                if w <= 0.0:
                    continue
                new_score = score + np.log(w + eps)
                new_path  = path + [nxt]
                candidates.append((new_score, new_path))
                completed.append(new_path)

        if not candidates:
            break

        # Keep top beam_width by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    if bridge is not None:
        completed = [p for p in completed if bridge in p]

    return completed


def sampled_paths(
    A          : np.ndarray,
    agent_ids  : list[str],
    origin     : str,
    Lp         : int,
    n_samples  : int = 200,
    bridge     : str | None = None,
    rng_seed   : int = 42,
) -> list[list[str]]:
    """
    LARGE scale: random-walk sampling from `origin`.
    Each walk follows edges with probability proportional to A[i,j]
    until length Lp or dead end.

    n_samples walks are generated; duplicates are kept (they get
    higher scores in the ranking step, which is the correct behaviour).
    """
    idx = {a: i for i, a in enumerate(agent_ids)}
    rng = np.random.default_rng(seed=rng_seed)
    paths: list[list[str]] = []

    for _ in range(n_samples):
        path = [origin]
        for _ in range(Lp - 1):
            last    = path[-1]
            li      = idx[last]
            row     = A[li].copy()
            # Zero out already-visited nodes (no cycles)
            for visited in path:
                row[idx[visited]] = 0.0
            total = row.sum()
            if total <= 0.0:
                break
            probs = row / total
            nxt   = rng.choice(agent_ids, p=probs)
            path.append(nxt)

        if len(path) >= 2:
            if bridge is None or bridge in path:
                paths.append(path)

    return paths


# ── Shared internal helper ─────────────────────────────────────────────────

def _matrix_to_nx(
    A         : np.ndarray,
    agent_ids : list[str],
    threshold : float = 0.0,
) -> nx.DiGraph:
    """
    Convert dense N×N weight matrix to a labelled networkx DiGraph.
    Only adds edges where A[i,j] > threshold.
    """
    G = nx.DiGraph()
    G.add_nodes_from(agent_ids)
    n = len(agent_ids)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > threshold:
                G.add_edge(agent_ids[i], agent_ids[j], weight=float(A[i, j]))
    return G
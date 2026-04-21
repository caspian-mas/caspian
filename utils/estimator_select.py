"""
caspian/utils/estimator_select.py

Practical CMI estimators for LI-CTE computation (step 7).

Spec says:
    Estimate I(U ; V | H) where U, V, H are interaction-restricted
    state vectors. The formal object is LI-CTE. The estimator is an
    implementation choice.

Three estimators, selected automatically by scale tier:
    SMALL  (N <= 10)  : kNN-CMI  — exact, sample-efficient
    MEDIUM (10 < N <= 50) : Gaussian-CMI — closed form, fast
    LARGE  (N > 50)   : LowRank-CMI — Gaussian-CMI on projected subspace

All estimators share the same interface:
    estimate(U, V, H) -> float >= 0

Inputs:
    U : np.ndarray shape (n_samples, d_u)  source state
    V : np.ndarray shape (n_samples, d_v)  target state
    H : np.ndarray shape (n_samples, d_h)  target history (conditioning)

Returns:
    float — estimated I(U ; V | H) in nats, clipped to >= 0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


# ── Estimator protocol ─────────────────────────────────────────────────────

@runtime_checkable
class CMIEstimator(Protocol):
    def estimate(
        self,
        U: np.ndarray,
        V: np.ndarray,
        H: np.ndarray,
    ) -> float: ...


# ── kNN-CMI (SMALL scale) ──────────────────────────────────────────────────

class KNNCMIEstimator:
    """
    Kraskov-style kNN conditional mutual information.

    I(U;V|H) estimated via:
        I(U;V|H) = ψ(k) - <ψ(n_UH)> - <ψ(n_VH)> + <ψ(n_H)>

    where ψ is the digamma function and n_XY is the number of points
    within the k-th neighbour distance in the joint space, projected
    to the XY subspace.

    Parameters
    ----------
    k : int
        Number of nearest neighbours. From SmallScaleConfig.knn_neighbors.
        Must be < n_samples - 1. Falls back to Gaussian if violated.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def estimate(
        self,
        U: np.ndarray,
        V: np.ndarray,
        H: np.ndarray,
    ) -> float:
        from sklearn.neighbors import NearestNeighbors
        from scipy.special import digamma

        U, V, H = _ensure_2d(U), _ensure_2d(V), _ensure_2d(H)
        n = U.shape[0]

        # Need at least k+2 samples for meaningful estimate
        if n < self.k + 2:
            return GaussianCMIEstimator().estimate(U, V, H)

        UVH = np.hstack([U, V, H])
        UH  = np.hstack([U, H])
        VH  = np.hstack([V, H])

        # Distance to k-th neighbour in joint space (Chebyshev = max norm)
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="chebyshev")
        nn.fit(UVH)
        dists, _ = nn.kneighbors(UVH)
        eps = dists[:, -1]  # shape (n,) — radius for each point

        n_UH = _count_within(UH,  eps)  # shape (n,)
        n_VH = _count_within(VH,  eps)
        n_H  = _count_within(H,   eps)

        # Kraskov estimator I (nats)
        cmi = (
            digamma(self.k)
            - np.mean(digamma(n_UH))
            - np.mean(digamma(n_VH))
            + np.mean(digamma(n_H))
        )
        return float(max(cmi, 0.0))


def _count_within(X: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    For each point i, count neighbours j (j != i) with
    Chebyshev distance <= eps[i]. Returns array of shape (n,), min 1.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(metric="chebyshev")
    nn.fit(X)
    counts = np.array([
        nn.radius_neighbors(
            X[i:i+1], radius=eps[i], return_distance=False
        )[0].shape[0] - 1          # subtract self
        for i in range(n)
    ])
    return np.maximum(counts, 1)   # digamma(0) is undefined


# ── Gaussian-CMI (MEDIUM scale) ────────────────────────────────────────────

class GaussianCMIEstimator:
    """
    Closed-form CMI under joint Gaussian assumption.

    I(U;V|H) = 0.5 * log( |Σ_UH| * |Σ_VH| / (|Σ_H| * |Σ_UVH|) )

    Fast and numerically stable for medium scale.
    Falls back to 0.0 on singular covariance (degenerate channel).

    No parameters — fully determined by the data.
    """

    def estimate(
        self,
        U: np.ndarray,
        V: np.ndarray,
        H: np.ndarray,
    ) -> float:
        U, V, H = _ensure_2d(U), _ensure_2d(V), _ensure_2d(H)

        if U.shape[0] < 2:
            return 0.0

        UVH = np.hstack([U, V, H])
        UH  = np.hstack([U, H])
        VH  = np.hstack([V, H])

        try:
            ld_uvh = _log_det_cov(UVH)
            ld_uh  = _log_det_cov(UH)
            ld_vh  = _log_det_cov(VH)
            ld_h   = _log_det_cov(H)
            cmi = 0.5 * (ld_uh + ld_vh - ld_h - ld_uvh)
            return float(max(cmi, 0.0))
        except np.linalg.LinAlgError:
            return 0.0


def _log_det_cov(X: np.ndarray) -> float:
    """
    Log-determinant of the sample covariance of X.
    Adds a small ridge (1e-6 * I) for numerical stability.
    Raises LinAlgError if still singular after ridge.
    """
    cov = np.cov(X.T)
    if cov.ndim == 0:               # scalar case (1-d input)
        cov = np.array([[float(cov)]])
    cov += 1e-6 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Non-positive-definite covariance")
    return float(logdet)


# ── LowRank-CMI (LARGE scale) ─────────────────────────────────────────────

class LowRankCMIEstimator:
    """
    Projects U, V, H to a low-dimensional subspace via randomized SVD,
    then applies Gaussian-CMI on the projections.

    Reduces O(d^2) covariance cost to O(rank^2) — tractable for large N
    where d (encoding dimension) can be large.

    Parameters
    ----------
    rank : int
        Projection dimension. From LargeScaleConfig.lowrank_dim.
        Capped at min(rank, n_samples-1, d) at runtime.
    """

    def __init__(self, rank: int = 16) -> None:
        self.rank = rank

    def estimate(
        self,
        U: np.ndarray,
        V: np.ndarray,
        H: np.ndarray,
    ) -> float:
        from sklearn.utils.extmath import randomized_svd

        U, V, H = _ensure_2d(U), _ensure_2d(V), _ensure_2d(H)
        n = U.shape[0]

        if n < 2:
            return 0.0

        U = _project(U, self.rank, randomized_svd)
        V = _project(V, self.rank, randomized_svd)
        H = _project(H, self.rank, randomized_svd)

        return GaussianCMIEstimator().estimate(U, V, H)


def _project(X: np.ndarray, rank: int, svd_fn) -> np.ndarray:
    """Project X onto its top-`rank` right singular vectors."""
    n, d = X.shape
    rank = min(rank, n - 1, d)
    if rank < 1:
        return X
    _, _, Vt = svd_fn(X, n_components=rank, random_state=42)
    return X @ Vt.T   # shape (n, rank)


# ── Factory ────────────────────────────────────────────────────────────────

def get_estimator(config) -> CMIEstimator:
    """
    Return the right CMI estimator for config's active scale tier.
    Requires config.resolve(n_agents) to have been called first.

    config : DetectorConfig
    """
    from caspian.utils.config import SmallScaleConfig, MediumScaleConfig, LargeScaleConfig

    sc = config.active_scale_config

    if isinstance(sc, SmallScaleConfig):
        return KNNCMIEstimator(k=sc.knn_neighbors)

    if isinstance(sc, MediumScaleConfig):
        return GaussianCMIEstimator()

    if isinstance(sc, LargeScaleConfig):
        return LowRankCMIEstimator(rank=sc.lowrank_dim)

    raise ValueError(f"Unknown scale config type: {type(sc)}")


# ── Shared helpers ─────────────────────────────────────────────────────────

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Guarantee shape (n, d) — promote 1-D arrays."""
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x
"""
caspian/utils/config.py

All monitor-resolution choices in one place.
Nothing here is a 'theorem' — these are calibration knobs.
See docs/config_reference.md for tuning guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from utils.types import ScaleTier, Topology


# ── Estimator configs per scale tier ───────────────────────────────────────

@dataclass
class SmallScaleConfig:
    """
    N <= 10.
    Exact kNN conditional MI, dense eigensolve, full path enumeration.
    """
    estimator        : Literal["knn_cmi"] = "knn_cmi"
    knn_neighbors    : int   = 5       # k for kNN-CMI
    eigen_method     : Literal["dense"] = "dense"
    max_spine_paths  : int   = 500     # enumerate up to this many candidate paths


@dataclass
class MediumScaleConfig:
    """
    10 < N <= 50.
    Gaussian CMI, sparse eigensolve (ARPACK), beam search for spines.
    """
    estimator        : Literal["gaussian_cmi"] = "gaussian_cmi"
    eigen_method     : Literal["sparse"] = "sparse"
    eigen_k          : int   = 6       # number of eigenvalues to compute
    beam_width       : int   = 20      # beam search width for spine inference


@dataclass
class LargeScaleConfig:
    """
    N > 50.
    Low-rank CMI approximation, power iteration for lambda1,
    sampled spine search.
    """
    estimator        : Literal["lowrank_cmi"] = "lowrank_cmi"
    lowrank_dim      : int   = 16      # projection dimension for low-rank CMI
    power_iter_steps : int   = 30      # iterations for lambda1 power method
    spine_samples    : int   = 200     # random path samples for spine ranking


# ── Main detector config ───────────────────────────────────────────────────

@dataclass
class DetectorConfig:
    """
    Single config object passed to CaspianDetector and threaded
    through the entire pipeline.

    Monitor-resolution choices (not model parameters):
      L   — lag horizon for LI-CTE (how far back influence is attributed)
      W   — rolling window for median/MAD self-normalization
      tau — onset threshold for Z(t) > tau
      mp  — persistence horizon (turns) for multi-step classification
      delta — onset attribution horizon (turns after t0 for origin scoring)
      K   — number of top spines to return
      Lp  — max path length for spine search

    Calibrated operating point:
      tau — should be set via scripts/calibrate_tau.py on held-out traces,
            not left at the default for production use.
    """

    # ── topology & scale ───────────────────────────────────────────────────
    topology         : Topology  = Topology.MESH
    # If None, auto-detected from |V| at init
    scale_tier       : ScaleTier | None = None

    # ── LI-CTE lag horizon ─────────────────────────────────────────────────
    # Largest lag over which the monitor attributes influence.
    # Keep small (1-3) in practice; larger values increase compute
    # and risk spurious attribution.
    L                : int   = 2

    # ── rolling self-normalization window ──────────────────────────────────
    # Number of past turns used to compute rolling median and MAD.
    # Too small → noisy normalization. Too large → slow to adapt.
    W                : int   = 20

    # ── onset detection threshold ──────────────────────────────────────────
    # Z(t) > tau triggers cascade onset.
    # Default 2.5 is a starting point — calibrate on your system.
    tau              : float = 2.5

    # ── cascade type classification ────────────────────────────────────────
    # Number of turns after onset over which elevated anomaly must
    # persist to be classified as multi-step.
    mp               : int   = 3

    # ── origin attribution horizon ─────────────────────────────────────────
    # Turns after t0 over which early outbound flow is scored.
    # Short by design — we want the *earliest* strong source.
    delta            : int   = 2

    # ── spine inference ────────────────────────────────────────────────────
    # K: number of top-scoring paths to return.
    K                : int   = 5
    # Lp: max path length. Auto-capped at min(cascade_length, graph_diameter).
    # Set None to always use the auto-cap.
    Lp               : int | None = None

    # ── numerical tolerances ───────────────────────────────────────────────
    # epsilon for channel matrix normalization denominator
    eps_norm         : float = 1e-8
    # epsilon_A for active edge threshold in cross-channel metric
    eps_active       : float = 1e-4
    # epsilon for onset score denominator (MAD stability)
    eps_mad          : float = 1e-8

    # ── scale-tier sub-configs ─────────────────────────────────────────────
    # Auto-populated by resolve() if not set manually.
    small  : SmallScaleConfig  = field(default_factory=SmallScaleConfig)
    medium : MediumScaleConfig = field(default_factory=MediumScaleConfig)
    large  : LargeScaleConfig  = field(default_factory=LargeScaleConfig)

    def resolve(self, n_agents: int) -> DetectorConfig:
        """
        Auto-detect scale tier from agent count if not already set.
        Returns self (mutates in place) for chaining.
        """
        if self.scale_tier is None:
            if n_agents <= 10:
                self.scale_tier = ScaleTier.SMALL
            elif n_agents <= 50:
                self.scale_tier = ScaleTier.MEDIUM
            else:
                self.scale_tier = ScaleTier.LARGE
        return self

    @property
    def active_scale_config(self) -> SmallScaleConfig | MediumScaleConfig | LargeScaleConfig:
        """Return the sub-config for the current scale tier."""
        if self.scale_tier is None:
            raise RuntimeError("Call resolve(n_agents) before accessing active_scale_config.")
        return {
            ScaleTier.SMALL:  self.small,
            ScaleTier.MEDIUM: self.medium,
            ScaleTier.LARGE:  self.large,
        }[self.scale_tier]


# ── Convenience presets ────────────────────────────────────────────────────
# Use these as starting points; always calibrate tau for your system.

def default_config(topology: Topology = Topology.MESH) -> DetectorConfig:
    """Balanced defaults. Good starting point for most systems."""
    return DetectorConfig(topology=topology)


def sensitive_config(topology: Topology = Topology.MESH) -> DetectorConfig:
    """
    Lower tau for high-recall detection.
    More false positives — use when missing a cascade is very costly.
    """
    return DetectorConfig(topology=topology, tau=1.5, W=15, L=3)


def precise_config(topology: Topology = Topology.MESH) -> DetectorConfig:
    """
    Higher tau for high-precision detection.
    Fewer false positives — use when alert fatigue is a concern.
    """
    return DetectorConfig(topology=topology, tau=3.5, W=30, L=1)
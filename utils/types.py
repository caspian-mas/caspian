"""
caspian/utils/types.py

All shared dataclasses and enums for CASPIAN.
Nothing in this file imports from any other caspian module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────────

class Channel(Enum):
    COMM = "comm"    # communication / messaging
    MEM  = "mem"     # memory / retrieval
    TOOL = "tool"    # tool-use / API calls
    EXEC = "exec"    # execution / control-flow / resources

    @property
    def basis_index(self) -> int:
        return list(Channel).index(self)


class Topology(Enum):
    MESH       = "mesh"
    HIERARCHY  = "hierarchy"
    PIPELINE   = "pipeline"
    STAR       = "star"
    HYBRID     = "hybrid"
    CUSTOM     = "custom"


class ScaleTier(Enum):
    SMALL  = "small"   # N <= 10   — exact kNN-CMI
    MEDIUM = "medium"  # 10 < N <= 50 — Gaussian-CMI
    LARGE  = "large"   # N > 50    — low-rank CMI


class CascadeType(Enum):
    SINGLE_STEP = "single_step"
    MULTI_STEP  = "multi_step"


# ── Channel state ──────────────────────────────────────────────────────────

@dataclass
class ChannelState:
    """
    Raw state object for one agent, one channel, one turn.
    Populated by adapter hooks; consumed by li_cte.py.
    """
    agent_id : str
    channel  : Channel
    turn     : int
    # Content relevant to this channel at this turn.
    # Format is channel-specific (see adapter hooks).
    payload  : list[Any] = field(default_factory=list)


@dataclass
class LateInteractionSupport:
    """
    The (U, V, H) triple for one directed edge, one channel, one turn.
    U  — subset of source state that could influence target
    V  — subset of target state reflecting that influence
    H  — target's own prior channel history (for conditioning)
    """
    src      : str
    tgt      : str
    channel  : Channel
    turn     : int
    U        : list[Any] = field(default_factory=list)
    V        : list[Any] = field(default_factory=list)
    H        : list[Any] = field(default_factory=list)


# ── Canonical interaction event ────────────────────────────────────────────

@dataclass
class InteractionEvent:
    """
    The normalized unit emitted by every adapter.
    One event = one agent acting at one turn across all channels.
    """
    agent_id      : str
    turn          : int
    comm_state    : ChannelState | None = None
    mem_state     : ChannelState | None = None
    tool_state    : ChannelState | None = None
    exec_state    : ChannelState | None = None
    # Raw metadata passthrough (framework-specific, not used by core)
    meta          : dict[str, Any] = field(default_factory=dict)

    def get_channel_state(self, channel: Channel) -> ChannelState | None:
        return {
            Channel.COMM: self.comm_state,
            Channel.MEM:  self.mem_state,
            Channel.TOOL: self.tool_state,
            Channel.EXEC: self.exec_state,
        }[channel]


# ── Per-turn influence ─────────────────────────────────────────────────────

@dataclass
class EdgeInfluence:
    """
    Fully computed influence for one directed edge at one turn.
    Output of fusion.py.
    """
    src              : str
    tgt              : str
    turn             : int
    # Per-channel LI-CTE values (length-4 array in Channel order)
    channel_values   : list[float]          # [comm, mem, tool, exec]
    fused_weight     : float                # L1 norm of channel-embedded vector
    dominant_channel : Channel


@dataclass
class TurnMetrics:
    """
    The three direct detection metrics for one turn.
    Output of metrics.py.
    """
    turn     : int
    lambda1  : float   # Perron root of A_t
    r        : float   # lambda2 / lambda1 (synchrony)
    p        : float   # cross-channel propagation proportion
    z_lambda : float   # robust normalized anomaly for lambda1
    z_r      : float   # robust normalized anomaly for r
    z_p      : float   # robust normalized anomaly for p
    Z        : float   # onset score = min(z_lambda, z_r, z_p)


# ── Cascade result ─────────────────────────────────────────────────────────

@dataclass
class Spine:
    """One candidate cascade propagation path."""
    nodes : list[str]
    score : float       # log-flow score


@dataclass
class CascadeResult:
    """
    Full output for one detected cascade event.
    This is the final object returned by CaspianDetector.
    """
    # Temporal window
    t0           : int
    t1           : int
    cascade_type : CascadeType

    # Attribution
    origin       : str
    bridge       : str | None
    amplifiers   : list[str]

    # Propagation paths
    spines       : list[Spine]          # ranked top-K

    # Supporting data
    turn_metrics : list[TurnMetrics]    # one per turn in [t0, t1]
    channel_mass : dict[str, list[float]]  # channel -> time series of M_c(t)

    # Passthrough
    meta         : dict[str, Any] = field(default_factory=dict)


# ── Benchmark types ────────────────────────────────────────────────────────

@dataclass
class GroundTruth:
    """Ground-truth label for one benchmark scenario."""
    scenario_id   : str
    has_cascade   : bool
    t0            : int | None = None
    t1            : int | None = None
    cascade_type  : CascadeType | None = None
    origin        : str | None = None
    bridge        : str | None = None
    amplifiers    : list[str] = field(default_factory=list)
    attack_vector : str | None = None   # e.g. "prompt_injection", "memory_poison"


@dataclass
class BenchmarkScenario:
    """One runnable scenario from any benchmark."""
    scenario_id   : str
    description   : str
    topology      : Topology
    n_agents      : int
    events        : list[InteractionEvent]
    ground_truth  : GroundTruth
    meta          : dict[str, Any] = field(default_factory=dict)
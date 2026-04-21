"""
caspian/core/channel_states.py

Steps 3 and 5 — Channel state construction and late-interaction support.

Spec step 5:
    For each agent i, channel c, turn t:
        S_i^(c)(t)  — the channel state object

    For each feasible edge i→j, channel c, turn t:
        U_ij^(c)(t) ⊆ S_i^(c)(t)  — subset of i's state that could influence j
        V_ij^(c)(t) ⊆ S_j^(c)(t)  — subset of j's state reflecting that influence
        H_j^(c)(t-1)               — j's own prior channel history (conditioning)

Spec step 5.1-5.4 define U and V concretely per channel:
    comm:  U = message chunks from i visible to j
           V = j's response attributable to upstream input
    mem:   U = memory entries written by i
           V = memory entries retrieved/reused by j
    tool:  U = tool suggestions, args, tool-linked evidence from i
           V = tool-call arguments/outputs used by j
    exec:  U = control signals, branch triggers from i
           V = downstream execution/branching state in j

Key constraint from spec:
    "We do not compute CTE over full agent hidden state.
     We compute it over the interaction-restricted state
     relevant to the specific channel and edge."

This file also encodes payloads to fixed-size numpy vectors so
li_cte.py can pass them to the CMI estimators.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from utils.types import (
    Channel,
    ChannelState,
    InteractionEvent,
    LateInteractionSupport,
)
from core.graph import StructuralGraph


# ── History store ──────────────────────────────────────────────────────────

class ChannelHistory:
    """
    Maintains per-agent per-channel history of ChannelState objects
    across turns. Used to construct H_j^(c)(t-1) — the conditioning
    term in LI-CTE.

    The spec says H is "j's own immediate channel history" — we keep
    the last L+1 turns so the LI-CTE engine can access any lag offset.
    """

    def __init__(self, max_len: int = 10) -> None:
        # agent_id -> channel -> list of ChannelState (oldest first)
        self._store: dict[str, dict[Channel, list[ChannelState]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._max_len = max_len

    def push(self, event: InteractionEvent) -> None:
        """Record all channel states from one InteractionEvent."""
        for channel in Channel:
            state = event.get_channel_state(channel)
            if state is not None:
                buf = self._store[event.agent_id][channel]
                buf.append(state)
                # Keep only the last max_len states
                if len(buf) > self._max_len:
                    buf.pop(0)

    def get_history(
        self,
        agent_id : str,
        channel  : Channel,
    ) -> list[ChannelState]:
        """
        Return all stored ChannelStates for agent/channel (oldest first).
        This is H_j^(c) — the agent's own past on this channel.
        """
        return list(self._store[agent_id][channel])

    def get_payload_at_lag(
        self,
        agent_id : str,
        channel  : Channel,
        lag      : int,
        current_turn : int,
    ) -> list[Any]:
        """
        Return the payload from (current_turn - lag) for agent/channel.
        Returns empty list if that turn is not in history.
        Used by li_cte.py to build lagged U matrices.
        """
        target_turn = current_turn - lag
        for state in self._store[agent_id][channel]:
            if state.turn == target_turn:
                return state.payload
        return []

    def reset(self) -> None:
        self._store.clear()


# ── Late-interaction support extraction ───────────────────────────────────

def extract_late_interaction_supports(
    events   : dict[str, InteractionEvent],   # agent_id -> event at turn t
    history  : ChannelHistory,                # history up to t-1
    graph    : StructuralGraph,
    channel  : Channel,
    turn     : int,
) -> dict[tuple[str, str], LateInteractionSupport]:
    """
    Step 5: For all feasible edges at turn t, extract (U, V, H) for
    the given channel.

    Parameters
    ----------
    events  : map of agent_id -> InteractionEvent for the current turn
    history : ChannelHistory holding states up to t-1
    graph   : StructuralGraph providing the feasible edge set E0
    channel : which of the four channels to process
    turn    : current turn index t

    Returns
    -------
    dict keyed by (src_id, tgt_id) -> LateInteractionSupport
    Only feasible edges with at least one non-empty payload are included.
    """
    supports: dict[tuple[str, str], LateInteractionSupport] = {}

    for src_id, tgt_id in graph.feasible_edges_named():
        src_event = events.get(src_id)
        tgt_event = events.get(tgt_id)

        # Both agents must have acted this turn to have a support
        if src_event is None or tgt_event is None:
            continue

        src_state = src_event.get_channel_state(channel)
        tgt_state = tgt_event.get_channel_state(channel)

        # U_ij^(c)(t): src's current channel payload
        # — "subset of i's state that could influence j"
        U = src_state.payload if src_state is not None else []

        # V_ij^(c)(t): tgt's current channel payload
        # — "subset of j's state reflecting that influence"
        V = tgt_state.payload if tgt_state is not None else []

        # H_j^(c)(t-1): tgt's own prior history on this channel
        # — the conditioning term that removes j's self-driven dynamics
        H = history.get_history(tgt_id, channel)
        H_payload = [item for s in H for item in s.payload]

        # Spec: we only need edges where there is something to measure.
        # Skip edges where both U and V are empty — no interaction occurred.
        if not U and not V:
            continue

        supports[(src_id, tgt_id)] = LateInteractionSupport(
            src     = src_id,
            tgt     = tgt_id,
            channel = channel,
            turn    = turn,
            U       = U,
            V       = V,
            H       = H_payload,
        )

    return supports


# ── Payload encoding ───────────────────────────────────────────────────────
# Converts raw payload lists (arbitrary Python objects from adapters)
# into fixed-size float32 numpy vectors for the CMI estimators.

ENCODE_DIM = 32   # default feature vector dimension


def encode_payload(payload: list[Any], dim: int = ENCODE_DIM) -> np.ndarray:
    """
    Encode a channel payload (list of arbitrary items) into a
    fixed-size float32 numpy vector of shape (dim,).

    Strategy per item type:
      - np.ndarray     : flatten, truncate/pad to dim
      - list/tuple of numbers : convert to array, truncate/pad
      - dict           : hash-encode key+value pairs, mean-pool
      - str            : hash-based bag encoding
      - int/float      : place at index 0, pad rest with 0
      - other          : str() then hash-encode

    Empty payload → zero vector.

    This is intentionally simple. Adapters that want richer
    encoding can pre-encode payloads as np.ndarrays — this
    function will pass them through directly.
    """
    if not payload:
        return np.zeros(dim, dtype=np.float32)

    item_vecs: list[np.ndarray] = []
    for item in payload:
        if isinstance(item, np.ndarray):
            v = item.flatten().astype(np.float32)

        elif isinstance(item, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in item):
                v = np.array(item, dtype=np.float32)
            else:
                v = _hash_encode(str(item), dim)

        elif isinstance(item, dict):
            parts = [
                _hash_encode(str(k) + "=" + str(val), dim)
                for k, val in item.items()
            ]
            v = np.mean(parts, axis=0).astype(np.float32) if parts else np.zeros(dim, dtype=np.float32)

        elif isinstance(item, str):
            v = _hash_encode(item, dim)

        elif isinstance(item, (int, float)):
            v = np.zeros(dim, dtype=np.float32)
            v[0] = float(item)

        else:
            v = _hash_encode(str(item), dim)

        # Truncate or zero-pad to exactly dim
        if len(v) >= dim:
            item_vecs.append(v[:dim])
        else:
            item_vecs.append(np.pad(v, (0, dim - len(v))))

    return np.mean(item_vecs, axis=0).astype(np.float32)


def encode_support(
    support : LateInteractionSupport,
    dim     : int = ENCODE_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a LateInteractionSupport into (U_vec, V_vec, H_vec).

    Returns shape (1, dim) arrays — single-sample row vectors.
    li_cte.py accumulates these across lags into (n_samples, dim)
    matrices before passing to the CMI estimator.
    """
    U = encode_payload(support.U, dim).reshape(1, dim)
    V = encode_payload(support.V, dim).reshape(1, dim)
    H = encode_payload(support.H, dim).reshape(1, dim)
    return U, V, H


# ── Internal helpers ───────────────────────────────────────────────────────

def _hash_encode(s: str, dim: int) -> np.ndarray:
    """
    Hash-based bag-of-characters encoding.
    Each character is scattered into a bucket via a simple hash,
    then the vector is L2-normalised.

    Deterministic and fast — no tokeniser needed.
    Characters beyond position 256 are ignored (long strings truncated).
    """
    v = np.zeros(dim, dtype=np.float32)
    for pos, ch in enumerate(s[:256]):
        # Two-prime scatter to reduce collisions
        bucket = (ord(ch) * 2654435761 + pos * 40503) % dim
        v[bucket] += 1.0
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v
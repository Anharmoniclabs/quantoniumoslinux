"""
Mimosa Pudica Gate — Dynamic Signal Cascade for Sequence Compression

Inspired by the mimosa pudica plant (sensitive plant) leaf-folding response.

Plant mechanics mapped to computation:
  - Electrical action potential  → fast threshold detection on activation norm
  - Hydraulic propagation        → cascade signal spreading to neighboring positions
  - Pulvini turgor collapse      → gating factor T ∈ [0,1] applied to hidden states
  - Ion-pump recovery            → exponential turgor restoration over time/position
  - Habituation memory           → diminishing sensitivity for frequently-triggered positions

Key behavioral property:
    "The thought propagates through the chain, closes in on itself,
     and contracts to contain the full thought."

This means: when a high-salience signal arrives at any position, it
propagates backward AND forward through the sequence. Positions that
receive the cascade fold (T → 0) unless they are themselves high-salience.
The result is a sparse set of "open" positions that form the essential
compressed representation of the chain.

Classes:
    MimosaGate          — stateless per-sequence gating
    MimosaRecurrent     — stateful token-by-token (autoregressive) gate
    MimosaKVCompressor  — KV-cache eviction using mimosa cascade
"""

import numpy as np
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Core: stateless sequence gate
# ---------------------------------------------------------------------------

class MimosaGate:
    """
    Applies mimosa-pudica gating to a full sequence of hidden states.

    Parameters
    ----------
    threshold : float
        Activation norm (relative to sequence mean) that triggers a fold.
        Think of this as the mechanical pressure needed to collapse a pulvinus.
    propagation_decay : float
        How much the hydraulic signal attenuates per position step (0–1).
        0.5 means the neighbor gets half the signal, next-neighbor gets 0.25.
    recovery_window : int
        Number of positions over which turgor recovers after a fold.
        Analogous to the 10–30 minute ion-pump recovery.
    habituation_rate : float
        Per-trigger reduction in sensitivity.  After many triggers a position
        becomes habituated (like a wind-accustomed mimosa leaf) and stops folding.
    max_propagation_hops : int
        How far the hydraulic signal travels in each direction.
    """

    def __init__(
        self,
        threshold: float = 1.5,
        propagation_decay: float = 0.6,
        recovery_window: int = 8,
        habituation_rate: float = 0.15,
        max_propagation_hops: int = 4,
    ):
        self.threshold = threshold
        self.propagation_decay = propagation_decay
        self.recovery_window = recovery_window
        self.habituation_rate = habituation_rate
        self.max_propagation_hops = max_propagation_hops

    def gate(
        self,
        hidden_states: np.ndarray,          # (seq_len, hidden_dim)
        sensitivity: Optional[np.ndarray] = None,  # (seq_len,) prior habituation
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        gated_states : (seq_len, hidden_dim)  — hidden_states * turgor
        turgor       : (seq_len,)             — gate values in [0, 1]
        new_sensitivity : (seq_len,)          — updated habituation state
        """
        seq_len, hidden_dim = hidden_states.shape

        # --- Sensitivity (habituation) state ---
        if sensitivity is None:
            sensitivity = np.ones(seq_len, dtype=np.float32)
        else:
            sensitivity = np.asarray(sensitivity, dtype=np.float32).copy()

        # --- Fast electrical signal: norm-based threshold ---
        norms = np.linalg.norm(hidden_states, axis=-1).astype(np.float32)  # (seq_len,)
        mean_norm = norms.mean() + 1e-8
        # Relative norm scaled by sensitivity (habituated positions less reactive)
        relative_activation = (norms / mean_norm) * sensitivity   # (seq_len,)
        triggered = relative_activation > self.threshold            # (seq_len,) bool

        # --- Slow hydraulic signal: propagation ---
        hydraulic = np.zeros(seq_len, dtype=np.float32)
        for i in range(seq_len):
            if triggered[i]:
                # Propagate forward and backward with decay
                for hop in range(self.max_propagation_hops + 1):
                    signal = self.propagation_decay ** hop
                    if i - hop >= 0:
                        hydraulic[i - hop] += signal
                    if hop > 0 and i + hop < seq_len:
                        hydraulic[i + hop] += signal

        # --- Turgor: combined electrical + hydraulic signal collapses turgor ---
        # Positions that are themselves high-activation are "stiff" (resist folding)
        # Positions hit only by propagation are "soft" (fold easily)
        stiffness = relative_activation  # high-activation positions resist folding
        fold_pressure = hydraulic - stiffness
        turgor = np.clip(1.0 - fold_pressure / (self.threshold * 2), 0.0, 1.0)
        # Triggered positions stay open (they ARE the signal source)
        turgor = np.where(triggered, np.clip(relative_activation / self.threshold, 0.5, 1.0), turgor)

        # --- Recovery: restore turgor for positions far from any trigger ---
        # Positions within recovery_window of a trigger remain folded
        # Positions outside recover toward 1.0
        trigger_positions = np.where(triggered)[0]
        if len(trigger_positions) > 0:
            dist_to_nearest_trigger = np.min(
                np.abs(np.arange(seq_len)[:, None] - trigger_positions[None, :]),
                axis=1
            ).astype(np.float32)
            recovery = np.clip(dist_to_nearest_trigger / self.recovery_window, 0.0, 1.0)
            turgor = np.maximum(turgor, recovery)
        else:
            turgor = np.ones(seq_len, dtype=np.float32)  # no triggers → all open

        # --- Habituation: reduce sensitivity of triggered positions ---
        new_sensitivity = np.where(
            triggered,
            np.maximum(sensitivity - self.habituation_rate, 0.1),
            np.minimum(sensitivity + self.habituation_rate * 0.3, 1.0),  # slow recovery
        )

        # --- Apply gate ---
        gated_states = hidden_states * turgor[:, np.newaxis]

        return gated_states.astype(np.float32), turgor, new_sensitivity


# ---------------------------------------------------------------------------
# Recurrent: token-by-token (autoregressive) mimosa gate
# ---------------------------------------------------------------------------

class MimosaRecurrent:
    """
    Stateful mimosa gate for autoregressive/streaming inference.

    Processes one token at a time.  Maintains:
      - turgor_history : turgor values for all past positions
      - sensitivity    : per-position habituation
      - hydraulic_carry: residual hydraulic signal from previous positions

    The "fold-and-contain" behavior emerges naturally: when a new high-salience
    token arrives, its hydraulic signal propagates BACKWARD through the stored
    history, retroactively closing (folding) the surrounding context and leaving
    only the essential load-bearing positions open.
    """

    def __init__(
        self,
        threshold: float = 1.5,
        propagation_decay: float = 0.6,
        max_propagation_hops: int = 4,
        habituation_rate: float = 0.15,
    ):
        self.threshold = threshold
        self.propagation_decay = propagation_decay
        self.max_propagation_hops = max_propagation_hops
        self.habituation_rate = habituation_rate

        self._position = 0
        self._norm_history: List[float] = []
        self._turgor_history: List[float] = []
        self._sensitivity_history: List[float] = []

    def reset(self):
        self._position = 0
        self._norm_history.clear()
        self._turgor_history.clear()
        self._sensitivity_history.clear()

    def step(self, hidden: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process one token hidden state.

        hidden : (hidden_dim,)
        Returns: (gated_hidden, turgor_value)
        """
        norm = float(np.linalg.norm(hidden))
        pos = self._position

        # Running mean norm for relative activation
        if len(self._norm_history) == 0:
            mean_norm = norm + 1e-8
        else:
            mean_norm = float(np.mean(self._norm_history)) + 1e-8

        sensitivity = self._sensitivity_history[pos] if pos < len(self._sensitivity_history) else 1.0
        relative_act = (norm / mean_norm) * sensitivity
        triggered = relative_act > self.threshold

        # Turgor for this position starts at 1.0 (open)
        turgor = 1.0

        # If triggered: propagate hydraulic signal backward through history
        if triggered:
            for hop in range(1, self.max_propagation_hops + 1):
                back_pos = pos - hop
                if back_pos < 0:
                    break
                signal = self.propagation_decay ** hop
                if back_pos < len(self._turgor_history):
                    # Fold backward positions unless they are themselves high-salience
                    back_norm = self._norm_history[back_pos]
                    back_relative = (back_norm / mean_norm) * (
                        self._sensitivity_history[back_pos] if back_pos < len(self._sensitivity_history) else 1.0
                    )
                    if back_relative <= self.threshold:
                        # Soft position: fold it
                        fold = signal / (self.threshold * 2)
                        self._turgor_history[back_pos] = max(0.0, self._turgor_history[back_pos] - fold)

        # Update histories
        self._norm_history.append(norm)
        self._turgor_history.append(turgor)
        self._sensitivity_history.append(
            max(0.1, sensitivity - self.habituation_rate) if triggered
            else min(1.0, sensitivity + self.habituation_rate * 0.3)
        )

        self._position += 1
        return (hidden * turgor).astype(np.float32), turgor


# ---------------------------------------------------------------------------
# KV Compressor: evict/compress KV cache using mimosa cascade
# ---------------------------------------------------------------------------

class MimosaKVCompressor:
    """
    KV-cache eviction policy using mimosa-pudica cascade logic.

    When the cache is full and a new token arrives:
    1. Compute turgor for all cached positions based on their key norms
    2. Positions with lowest turgor (folded) are evicted first
    3. The new token is inserted at full turgor (T=1)

    This preserves the "high-salience anchor" positions (open leaves) while
    evicting the "low-information middle" (folded leaves) — analogous to how
    the plant collapses redundant leaf segments while the stem stays rigid.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of (K, V) pairs to retain.
    gate : MimosaGate
        The gating algorithm to use.
    """

    def __init__(self, max_cache_size: int = 512, gate: Optional[MimosaGate] = None):
        self.max_cache_size = max_cache_size
        self.gate = gate or MimosaGate(threshold=1.5, propagation_decay=0.6)
        self._keys: Optional[np.ndarray] = None     # (n_cached, n_kv, head_dim)
        self._values: Optional[np.ndarray] = None
        self._sensitivity: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return 0 if self._keys is None else self._keys.shape[0]

    def push(
        self,
        new_k: np.ndarray,  # (n_kv, head_dim)
        new_v: np.ndarray,  # (n_kv, head_dim)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add a new KV pair to the cache, evicting the lowest-turgor entry if full.

        Returns (k_cache, v_cache) — the full cache after insertion.
        """
        new_k = new_k[np.newaxis]  # (1, n_kv, head_dim)
        new_v = new_v[np.newaxis]

        if self._keys is None:
            self._keys = new_k
            self._values = new_v
            self._sensitivity = np.ones(1, dtype=np.float32)
            return self._keys, self._values

        # Append
        self._keys = np.concatenate([self._keys, new_k], axis=0)
        self._values = np.concatenate([self._values, new_v], axis=0)
        self._sensitivity = np.append(self._sensitivity, 1.0)

        if self.size > self.max_cache_size:
            # Compute turgor from key norms
            # Use mean of K heads as the "hidden state" representation
            key_repr = self._keys.mean(axis=1)  # (n_cached, head_dim)
            _, turgor, self._sensitivity = self.gate.gate(key_repr, self._sensitivity)
            # Evict lowest-turgor position (excluding the most recent token)
            evict_candidates = turgor[:-1]  # never evict the newest token
            evict_idx = int(np.argmin(evict_candidates))
            keep = [i for i in range(self.size) if i != evict_idx]
            self._keys = self._keys[keep]
            self._values = self._values[keep]
            self._sensitivity = self._sensitivity[keep]

        return self._keys, self._values

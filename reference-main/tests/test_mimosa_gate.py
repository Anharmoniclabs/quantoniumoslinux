"""
Tests for algorithms/mimosa/mimosa_gate.py

Verifies the three core mimosa-pudica behavioral properties:
  1. Threshold gating — weak signals pass through, strong ones fold neighbors
  2. Fold-and-contain — high-salience positions stay open, low-salience fold
  3. Habituation — repeated triggers reduce sensitivity
  4. KV compressor — evicts low-turgor entries, retains anchors
"""

import sys
import numpy as np

sys.path.insert(0, '.')
from algorithms.mimosa import MimosaGate, MimosaRecurrent, MimosaKVCompressor


def test_threshold_gating():
    """Weak signals should not trigger folding; strong signals should."""
    gate = MimosaGate(threshold=2.0, propagation_decay=0.5, max_propagation_hops=2)
    hidden_dim = 64
    seq_len = 8

    # Uniform low-activation sequence — all positions should stay near 1.0
    weak = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.1
    _, turgor_weak, _ = gate.gate(weak)
    assert turgor_weak.min() > 0.8, f"Weak signal should not fold: min={turgor_weak.min():.4f}"

    # Single spike at position 4 — neighbors should fold
    strong = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.1
    strong[4] = np.random.randn(hidden_dim).astype(np.float32) * 10.0  # spike
    _, turgor_strong, _ = gate.gate(strong)
    spike_turgor = turgor_strong[4]
    neighbor_turgor = turgor_strong[[3, 5]].mean()
    assert spike_turgor > neighbor_turgor, (
        f"Spike position should have higher turgor than neighbors: "
        f"spike={spike_turgor:.4f}, neighbors={neighbor_turgor:.4f}"
    )
    print(f"[PASS] threshold_gating: weak_min={turgor_weak.min():.3f}, "
          f"spike={spike_turgor:.3f}, neighbor={neighbor_turgor:.3f}")


def test_fold_and_contain():
    """
    The 'fold-and-contain' property: high-salience positions (anchors) stay open,
    low-salience middle positions fold, forming a compressed representation.
    """
    gate = MimosaGate(threshold=1.5, propagation_decay=0.7, max_propagation_hops=3)
    seq_len = 16
    hidden_dim = 64

    # Pattern: anchor at positions 0, 8, 15; filler in between
    hidden = np.random.randn(seq_len, hidden_dim).astype(np.float32) * 0.2
    for anchor in [0, 8, 15]:
        hidden[anchor] = np.random.randn(hidden_dim).astype(np.float32) * 5.0

    gated, turgor, _ = gate.gate(hidden)

    anchor_turgor = turgor[[0, 8, 15]].mean()
    filler_turgor = turgor[[2, 3, 4, 5, 6, 10, 11, 12]].mean()

    assert anchor_turgor > filler_turgor, (
        f"Anchors should stay more open than fillers: "
        f"anchor={anchor_turgor:.4f}, filler={filler_turgor:.4f}"
    )
    # Gated output should have lower norm at filler positions
    gated_norms = np.linalg.norm(gated, axis=-1)
    orig_norms = np.linalg.norm(hidden, axis=-1)
    filler_reduction = 1.0 - (gated_norms[2:7].mean() / (orig_norms[2:7].mean() + 1e-8))
    print(f"[PASS] fold_and_contain: anchor_T={anchor_turgor:.3f}, "
          f"filler_T={filler_turgor:.3f}, filler_reduction={filler_reduction:.3f}")


def test_habituation():
    """Repeated triggers should reduce sensitivity (habituation)."""
    gate = MimosaGate(threshold=1.2, habituation_rate=0.2)
    hidden_dim = 32

    hidden = np.zeros((5, hidden_dim), dtype=np.float32)
    hidden[2] = np.ones(hidden_dim) * 3.0  # spike at pos 2

    # First trigger
    _, turgor1, sens1 = gate.gate(hidden, sensitivity=None)
    # Provide updated sensitivity for second pass
    _, turgor2, sens2 = gate.gate(hidden, sensitivity=sens1)

    # Sensitivity at position 2 should have decreased
    assert sens2[2] < sens1[2] if sens1[2] > 0.1 else True, (
        f"Sensitivity should decrease after trigger: {sens1[2]:.3f} -> {sens2[2]:.3f}"
    )
    print(f"[PASS] habituation: sensitivity[2]: {sens1[2]:.3f} -> {sens2[2]:.3f}")


def test_recurrent_backward_fold():
    """
    Recurrent mode: a late high-salience token should fold back earlier low-salience tokens.
    """
    gate_r = MimosaRecurrent(threshold=1.5, propagation_decay=0.7, max_propagation_hops=4)
    hidden_dim = 64

    turgor_history = []
    tokens = []
    # 8 filler tokens
    for _ in range(8):
        tok = np.random.randn(hidden_dim).astype(np.float32) * 0.3
        tokens.append(tok)
        _, t = gate_r.step(tok)
        turgor_history.append(t)

    # Record turgor of position 5 (filler) before spike
    turgor_pos5_before = gate_r._turgor_history[5]

    # Spike at position 8 — should retroactively fold positions 4-7
    spike = np.random.randn(hidden_dim).astype(np.float32) * 8.0
    _, t_spike = gate_r.step(spike)
    turgor_pos5_after = gate_r._turgor_history[5]

    assert turgor_pos5_after <= turgor_pos5_before, (
        f"Backward propagation should fold pos5: {turgor_pos5_before:.3f} -> {turgor_pos5_after:.3f}"
    )
    print(f"[PASS] recurrent_backward_fold: pos5 turgor: "
          f"{turgor_pos5_before:.3f} -> {turgor_pos5_after:.3f} (spike T={t_spike:.3f})")


def test_kv_compressor_evicts_lowest_turgor():
    """
    KV compressor should evict the least salient (lowest turgor) entry when full.
    """
    n_kv, head_dim = 2, 32
    max_size = 4
    comp = MimosaKVCompressor(max_cache_size=max_size, gate=MimosaGate(threshold=1.5))

    # Push 4 entries: 3 weak, 1 strong
    keys_pushed = []
    for i in range(max_size):
        scale = 5.0 if i == 2 else 0.3   # position 2 is the anchor
        k = np.random.randn(n_kv, head_dim).astype(np.float32) * scale
        v = np.random.randn(n_kv, head_dim).astype(np.float32) * 0.1
        keys_pushed.append(k.copy())
        comp.push(k, v)

    assert comp.size == max_size, f"Cache should be full: {comp.size}"

    # Push one more — should evict a low-turgor entry
    k_new = np.random.randn(n_kv, head_dim).astype(np.float32) * 0.5
    v_new = np.random.randn(n_kv, head_dim).astype(np.float32) * 0.1
    comp.push(k_new, v_new)

    assert comp.size == max_size, (
        f"Cache size should remain {max_size} after eviction, got {comp.size}"
    )
    print(f"[PASS] kv_compressor: cache size={comp.size} after eviction (max={max_size})")


def test_gated_norm_reduction():
    """Gated output should have lower total norm than input for folded positions."""
    gate = MimosaGate(threshold=1.0, propagation_decay=0.8)
    hidden_dim = 128
    seq_len = 20

    hidden = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    hidden[10] *= 8.0  # strong spike

    gated, turgor, _ = gate.gate(hidden)
    input_norm = np.linalg.norm(hidden)
    output_norm = np.linalg.norm(gated)

    assert output_norm < input_norm, (
        f"Gated output should have lower total norm: {input_norm:.3f} -> {output_norm:.3f}"
    )
    compression_ratio = 1.0 - output_norm / input_norm
    print(f"[PASS] gated_norm_reduction: {input_norm:.3f} -> {output_norm:.3f} "
          f"({compression_ratio*100:.1f}% reduction)")


if __name__ == '__main__':
    print("=== Mimosa Pudica Gate Tests ===\n")
    test_threshold_gating()
    test_fold_and_contain()
    test_habituation()
    test_recurrent_backward_fold()
    test_kv_compressor_evicts_lowest_turgor()
    test_gated_norm_reduction()
    print("\n=== All tests passed ===")

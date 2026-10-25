import numpy as np

from quantonium_os_src.engine.RFTMW import MiddlewareTransformEngine, PHI


def test_wave_fibonacci_pruned_is_listed():
    engine = MiddlewareTransformEngine()
    variants = engine.list_all_variants()
    assert "wave_fibonacci_pruned" in variants
    assert engine.get_variant_info("wave_fibonacci_pruned") is not None


def test_wave_fibonacci_pruned_keeps_sparse_waveform():
    engine = MiddlewareTransformEngine()
    engine.selected_variant = "wave_fibonacci_pruned"

    n = 256
    samples = np.arange(n, dtype=float)
    x_bytes = ((np.sin(2 * np.pi * PHI * samples / n) * 127 + 127).clip(0, 255).astype(np.uint8)).tobytes()

    waveform = engine.binary_to_waveform(x_bytes)
    nnz = int(np.count_nonzero(np.abs(waveform) > 1e-12))
    max_expected = len(engine._wave_fibonacci_candidate_bins(n))

    assert nnz <= max_expected
    assert nnz < n // 4


def test_wave_fibonacci_pruned_round_trip_is_reasonable_for_phi_signal():
    engine = MiddlewareTransformEngine()
    engine.selected_variant = "wave_fibonacci_pruned"

    n = 256
    samples = np.arange(n, dtype=float)
    source = ((np.sin(2 * np.pi * PHI * samples / n) * 127 + 127).clip(0, 255).astype(np.uint8))

    waveform = engine.binary_to_waveform(source.tobytes())
    recovered = np.frombuffer(engine.waveform_to_binary(waveform, n), dtype=np.uint8)[:n]

    mae = float(np.mean(np.abs(recovered.astype(np.int16) - source.astype(np.int16))))
    assert mae < 35.0

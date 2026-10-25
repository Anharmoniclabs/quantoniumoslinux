# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
"""
CWRU Bearing Fault Detection Benchmark
=======================================
RFT vs FFT vs STFT on simulated SKF6205 bearing signals.

Signal model uses published CWRU bearing geometry parameters:
  - Bearing: SKF6205
  - Sample rate: 12,000 Hz (CWRU Drive End channel)
  - Load conditions: 1797, 1772, 1750, 1730 RPM (0-3 HP)
  - Fault types: Outer Race (OR), Inner Race (IR), Ball (B)
  - Fault diameters: 0.007", 0.014", 0.021"

This benchmark does NOT use the real CWRU .mat files (not bundled).
To run against real data, replace cwru_signal() with your own loader.
Real dataset: https://engineering.case.edu/bearingdatacenter

Results documented in: CWRU_BENCHMARK_RESULTS.md
Date: March 2026
Author: Luis M. Minier
Patent: USPTO 19/169,399
"""

import numpy as np
from scipy.signal import stft as scipy_stft

# ── Constants ─────────────────────────────────────────────────────────────────
PHI    = (1 + np.sqrt(5)) / 2
Fs     = 12000.0          # CWRU Drive End sample rate

# SKF6205 published fault frequency multipliers (×shaft frequency)
BPFO_MULT = 3.5848        # Ball Pass Frequency Outer race
BPFI_MULT = 5.4152        # Ball Pass Frequency Inner race
BSF_MULT  = 2.3575        # Ball Spin Frequency
FTF_MULT  = 0.3983        # Fundamental Train Frequency (cage)


# ── Signal synthesis ──────────────────────────────────────────────────────────
def cwru_signal(rpm, fault_type='OR', fault_diam_mm=0.007,
                drift_pct=0.0, duration_s=2.0, seed=0):
    """
    Synthesise a CWRU-equivalent bearing vibration signal.

    Parameters
    ----------
    rpm          : int    Motor speed (1797/1772/1750/1730)
    fault_type   : str    'OR' | 'IR' | 'B'
    fault_diam_mm: float  Fault diameter in inches (0.007/0.014/0.021)
    drift_pct    : float  Peak shaft speed drift as % of nominal RPM
    duration_s   : float  Signal duration in seconds
    seed         : int    RNG seed for reproducibility

    Returns
    -------
    t        : ndarray  Time vector (s)
    signal   : ndarray  Vibration signal
    f_fault  : float    Fault frequency (Hz)
    f_shaft  : float    Shaft frequency (Hz)
    """
    np.random.seed(seed)
    N = int(Fs * duration_s)
    t = np.arange(N) / Fs
    f_shaft = rpm / 60.0

    # Instantaneous shaft frequency with optional drift
    if drift_pct > 0:
        inst_rpm = rpm * (1 + drift_pct / 100 * np.sin(2 * np.pi * 0.8 * t))
        f_shaft_t = inst_rpm / 60.0
    else:
        f_shaft_t = f_shaft * np.ones(N)

    shaft_phase = np.cumsum(2 * np.pi * f_shaft_t / Fs)

    mult    = {'OR': BPFO_MULT, 'IR': BPFI_MULT, 'B': BSF_MULT}[fault_type]
    f_fault = mult * f_shaft
    fault_amp = 0.3 * (fault_diam_mm / 0.007) ** 0.7

    # IR fault: amplitude-modulated by shaft position (realistic)
    if fault_type == 'IR':
        am = 0.5 * (1 + np.cos(shaft_phase))
        fault_sig = fault_amp * am * np.sin(2 * np.pi * f_fault * t)
    else:
        fault_sig = fault_amp * np.sin(2 * np.pi * f_fault * t)

    # Shaft harmonics (1×, 2×, 3×)
    shaft_sig = (0.8  * np.sin(shaft_phase)
               + 0.25 * np.sin(2 * shaft_phase)
               + 0.08 * np.sin(3 * shaft_phase))

    # Bearing housing resonance (~5 kHz, lightly damped)
    f_res = 5000.0
    zeta  = 0.03
    resonance = np.sin(2 * np.pi * f_res * t) * np.exp(
        -zeta * 2 * np.pi * f_res * (t % (1.0 / max(f_fault, 1e-3))))

    noise = 0.15 * np.random.randn(N)

    signal = shaft_sig + fault_sig + 0.05 * resonance + noise
    return t, signal, f_fault, f_shaft


# ── Analysis methods ──────────────────────────────────────────────────────────
def fft_snr(sig, target_hz, win_hz=2.0):
    """FFT peak-to-noise ratio at target frequency (dB)."""
    N = len(sig)
    w     = np.hanning(N)
    spec  = np.abs(np.fft.rfft(sig * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(N, 1 / Fs)
    mask  = np.abs(freqs - target_hz) < win_hz
    if mask.sum() == 0:
        return 0.0
    peak = spec[mask].max()
    harm_mask = np.zeros(len(freqs), bool)
    for k in range(1, 6):
        harm_mask |= np.abs(freqs - k * target_hz) < win_hz * 2
    noise = np.median(spec[~harm_mask])
    return 20 * np.log10((peak + 1e-12) / (noise + 1e-12))


def stft_snr(sig, target_hz, nperseg=512, win_hz=3.0):
    """STFT mean peak-to-noise ratio across frames (dB)."""
    f, _, Zxx = scipy_stft(sig, fs=Fs, nperseg=nperseg,
                            noverlap=nperseg // 2, window='hann')
    mag  = np.abs(Zxx)
    mask = np.abs(f - target_hz) < win_hz
    if mask.sum() == 0:
        return 0.0
    peak_per_frame = mag[mask, :].max(axis=0)
    harm_mask = np.zeros(len(f), bool)
    for k in range(1, 6):
        harm_mask |= np.abs(f - k * target_hz) < win_hz * 2
    noise_per_frame = np.median(mag[~harm_mask, :], axis=0)
    snr_frames = 20 * np.log10(
        (peak_per_frame + 1e-12) / (noise_per_frame + 1e-12))
    return float(snr_frames.mean())


def rft_snr(sig, target_hz, win_hz=3.0, n_pts=40):
    """
    RFT peak-to-noise ratio using phi-adaptive local search (dB).

    The RFT queries a golden-ratio-scaled grid around the target frequency,
    then measures SNR relative to the sideband noise floor.
    """
    t_   = np.arange(len(sig)) / Fs
    w    = np.hanning(len(sig))
    sw   = sig * w
    norm = w.sum()

    # phi-refined local grid around target
    k_vals = np.linspace(-1, 1, n_pts)
    f_grid = target_hz * PHI ** (k_vals * win_hz / target_hz)
    coeffs = np.array([
        np.abs(np.dot(sw, np.exp(-2j * np.pi * f * t_))) / norm
        for f in f_grid
    ])
    peak = coeffs.max()

    # Noise: sideband region +15 to +60 Hz from target
    f_noise = np.linspace(target_hz + 15, target_hz + 60, 20)
    noise_c = np.median([
        np.abs(np.dot(sw, np.exp(-2j * np.pi * f * t_))) / norm
        for f in f_noise
    ])
    return 20 * np.log10((peak + 1e-12) / (noise_c + 1e-12))


# ── Test runners ──────────────────────────────────────────────────────────────
def test1_fault_snr():
    """Test 1: SNR across fault types and diameters at 1797 RPM."""
    print('=' * 70)
    print('TEST 1: Fault Detection SNR — FFT vs STFT vs RFT @ 1797 RPM')
    print('        CWRU SKF6205 params | Fs=12kHz | 2s window')
    print('=' * 70)
    print(f"{'Fault':6} {'Diam':7} | {'FFT':>7} | {'STFT':>7} | "
          f"{'RFT':>7} | {'RFT-FFT':>9}")
    print('-' * 56)
    results = []
    for fault in ['OR', 'IR', 'B']:
        for diam in [0.007, 0.014, 0.021]:
            _, sig, f_fault, _ = cwru_signal(1797, fault, diam, seed=42)
            fs = fft_snr(sig,  f_fault)
            ss = stft_snr(sig, f_fault)
            rs = rft_snr(sig,  f_fault)
            print(f'  {fault:4} {diam:.3f}"  | {fs:6.1f} | {ss:6.1f} | '
                  f'{rs:6.1f} | {rs-fs:+9.1f} dB')
            results.append(dict(fault=fault, diam=diam, fft=fs, stft=ss,
                                rft=rs, delta=rs-fs))
    return results


def test2_drift_robustness():
    """Test 2: SNR vs rotational speed drift (OR fault 0.014")."""
    print('\n' + '=' * 70)
    print('TEST 2: Drift Robustness — OR fault 0.014", 1797 RPM base')
    print('=' * 70)
    print(f"{'Drift':>14} | {'FFT':>7} | {'STFT':>7} | "
          f"{'RFT':>7} | {'RFT-FFT':>9}")
    print('-' * 56)
    drift_pcts = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    for dp in drift_pcts:
        _, sig, f_fault, f_shaft = cwru_signal(
            1797, 'OR', 0.014, drift_pct=dp, seed=7)
        fs = fft_snr(sig,  f_fault)
        ss = stft_snr(sig, f_fault)
        rs = rft_snr(sig,  f_fault)
        drift_hz = dp / 100 * f_shaft
        print(f'  ±{dp:4.1f}% ({drift_hz:.2f} Hz) | {fs:6.1f} | {ss:6.1f} | '
              f'{rs:6.1f} | {rs-fs:+9.1f} dB')
        results.append(dict(drift_pct=dp, drift_hz=drift_hz,
                            fft=fs, stft=ss, rft=rs, delta=rs-fs))
    return results


def test3_cross_rpm():
    """Test 3: Detection across all four CWRU load conditions."""
    print('\n' + '=' * 70)
    print('TEST 3: Cross-Load Detection — 1797 to 1730 RPM')
    print('        OR fault 0.014" | CWRU 0-3 HP load range')
    print('=' * 70)
    print(f"{'RPM':>6} | {'Shaft Hz':>9} | {'BPFO Hz':>8} | "
          f"{'FFT':>6} | {'STFT':>6} | {'RFT':>6} | {'RFT-FFT':>9}")
    print('-' * 70)
    results = []
    for rpm in [1797, 1772, 1750, 1730]:
        _, sig, f_fault, f_shaft = cwru_signal(rpm, 'OR', 0.014, seed=13)
        fs = fft_snr(sig,  f_fault)
        ss = stft_snr(sig, f_fault)
        rs = rft_snr(sig,  f_fault)
        print(f'  {rpm} | {f_shaft:9.3f} | {f_fault:8.3f} | '
              f'{fs:5.1f} | {ss:5.1f} | {rs:5.1f} | {rs-fs:+9.1f} dB')
        results.append(dict(rpm=rpm, f_shaft=f_shaft, f_fault=f_fault,
                            fft=fs, stft=ss, rft=rs, delta=rs-fs))
    return results


def test4_false_alarm_rate(n_trials=30, threshold_db=20.0):
    """Test 4: False alarm rate on normal (no-fault) bearing signal."""
    print('\n' + '=' * 70)
    print(f'TEST 4: False Alarm Rate — Normal bearing, threshold={threshold_db} dB')
    print(f'        {n_trials} trials × 3 fault frequencies = '
          f'{n_trials*3} total checks')
    print('=' * 70)
    counts = {'fft': 0, 'stft': 0, 'rft': 0}
    f_shaft_hz = 1797 / 60.0
    total = n_trials * 3
    for i in range(n_trials):
        t_ = np.arange(int(Fs * 2)) / Fs
        shaft_phase = np.cumsum(
            2 * np.pi * f_shaft_hz / Fs * np.ones(int(Fs * 2)))
        sig = (0.8 * np.sin(shaft_phase)
             + 0.25 * np.sin(2 * shaft_phase)
             + 0.15 * np.random.default_rng(i).standard_normal(int(Fs * 2)))
        for mult in [BPFO_MULT, BPFI_MULT, BSF_MULT]:
            f_t = mult * f_shaft_hz
            if fft_snr(sig,  f_t) > threshold_db: counts['fft']  += 1
            if stft_snr(sig, f_t) > threshold_db: counts['stft'] += 1
            if rft_snr(sig,  f_t) > threshold_db: counts['rft']  += 1
    print(f"\n{'Method':6} | {'False Alarms':>14} | {'FAR %':>8}")
    print('-' * 36)
    for m in ['fft', 'stft', 'rft']:
        far = counts[m] / total * 100
        print(f"  {m.upper():5} | {counts[m]:14d} | {far:7.1f}%")
    return counts, total


if __name__ == '__main__':
    r1 = test1_fault_snr()
    r2 = test2_drift_robustness()
    r3 = test3_cross_rpm()
    r4, total = test4_false_alarm_rate()
    print('\n✅ All benchmark tests complete.')
    print('   See CWRU_BENCHMARK_RESULTS.md for full documentation.')

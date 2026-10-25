# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
"""
cwru_benchmark_real.py
=======================
RFT vs FFT vs STFT on REAL CWRU .mat bearing recordings.

Requirements
------------
1. Download CWRU .mat files:
   https://engineering.case.edu/bearingdatacenter/download-data-file

2. Place them in the same folder as this script using standard CWRU numeric
   filenames (e.g. 97.mat, 130.mat). Do NOT rename the files.

   Fault files     : 105, 106, 118, 119, 130-133, 169-170, 185-186,
                     197-200, 209-210, 222-223, 234-235
   Normal baseline : 97, 98, 99, 100  <- required for FAR test

3. Run from repo root (bash):
   python algorithms/rft/applications/vibration/cwru_benchmark_real.py \\
     algorithms/rft/applications/vibration/

   Windows PowerShell:
   python algorithms/rft/applications/vibration/cwru_benchmark_real.py `
     algorithms/rft/applications/vibration/

For the synthetic benchmark (no download needed):
   python algorithms/rft/applications/vibration/cwru_benchmark.py

Author : Luis M. Minier
Patent : USPTO 19/169,399
Date   : March 2026
"""

import sys
import numpy as np
from scipy.signal import stft as scipy_stft
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2
FS        = 12000.0
BPFO_MULT = 3.5848
BPFI_MULT = 5.4152
BSF_MULT  = 2.3575
RPM       = 1797
F_SHAFT   = RPM / 60.0

FAULT_MAP = {
    'OR_0007': ('OR', 0.007, BPFO_MULT),
    'OR_0014': ('OR', 0.014, BPFO_MULT),
    'OR_0021': ('OR', 0.021, BPFO_MULT),
    'IR_0007': ('IR', 0.007, BPFI_MULT),
    'IR_0014': ('IR', 0.014, BPFI_MULT),
    'IR_0021': ('IR', 0.021, BPFI_MULT),
    'B_0007':  ('B',  0.007, BSF_MULT),
    'B_0014':  ('B',  0.014, BSF_MULT),
    'B_0021':  ('B',  0.021, BSF_MULT),
}


# ── SNR methods (identical to cwru_benchmark.py) ──────────────────────────────
def fft_snr(sig, target_hz, win_hz=2.0):
    N = len(sig)
    w    = np.hanning(N)
    spec = np.abs(np.fft.rfft(sig * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(N, 1 / FS)
    mask = np.abs(freqs - target_hz) < win_hz
    if not mask.any():
        return 0.0
    peak = spec[mask].max()
    harm_mask = np.zeros(len(freqs), bool)
    for k in range(1, 6):
        harm_mask |= np.abs(freqs - k * target_hz) < win_hz * 2
    noise = np.median(spec[~harm_mask])
    return 20 * np.log10((peak + 1e-12) / (noise + 1e-12))


def stft_snr(sig, target_hz, nperseg=512, win_hz=3.0):
    f, _, Zxx = scipy_stft(sig, fs=FS, nperseg=nperseg,
                            noverlap=nperseg // 2, window='hann')
    mag  = np.abs(Zxx)
    mask = np.abs(f - target_hz) < win_hz
    if not mask.any():
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
    t_  = np.arange(len(sig)) / FS
    w   = np.hanning(len(sig))
    sw  = sig * w
    norm = w.sum()
    k_vals = np.linspace(-1, 1, n_pts)
    f_grid = target_hz * PHI ** (k_vals * win_hz / target_hz)
    coeffs = np.array([
        np.abs(np.dot(sw, np.exp(-2j * np.pi * f * t_))) / norm
        for f in f_grid
    ])
    peak = coeffs.max()
    f_noise = np.linspace(target_hz + 15, target_hz + 60, 20)
    noise_c = np.median([
        np.abs(np.dot(sw, np.exp(-2j * np.pi * f * t_))) / norm
        for f in f_noise
    ])
    return 20 * np.log10((peak + 1e-12) / (noise_c + 1e-12))


# ── Main benchmark ───────────────────────────────────────────────────────────────────
def run_real_benchmark(data_dir: str):
    from .cwru_real_loader import load_cwru_dataset

    print('=' * 70)
    print('CWRU REAL DATA BENCHMARK — RFT vs FFT vs STFT')
    print(f'Data directory: {data_dir}')
    print(f'Bearing: SKF6205 | Fs={FS:.0f} Hz | Shaft={F_SHAFT:.3f} Hz')
    print('=' * 70)

    dataset = load_cwru_dataset(data_dir, duration_s=2.0, verbose=True)
    print()

    # ── Fault detection SNR ────────────────────────────────────────────────────
    print('=' * 70)
    print('FAULT DETECTION SNR — Real Recordings')
    print('=' * 70)
    print(f"{'Fault':6} {'Diam':7} | {'FFT':>7} | {'STFT':>7} | "
          f"{'RFT':>7} | {'RFT-FFT':>9}")
    print('-' * 56)

    results = []
    for label, (fault_type, diam, mult) in FAULT_MAP.items():
        if label not in dataset:
            print(f'  {fault_type:4} {diam:.3f}"  |  [file not found - skipping]')
            continue
        f_fault = mult * F_SHAFT
        fft_scores, stft_scores, rft_scores = [], [], []
        for sig in dataset[label]:
            fft_scores.append(fft_snr(sig,  f_fault))
            stft_scores.append(stft_snr(sig, f_fault))
            rft_scores.append(rft_snr(sig,  f_fault))
        fs = np.mean(fft_scores)
        ss = np.mean(stft_scores)
        rs = np.mean(rft_scores)
        n  = len(dataset[label])
        print(f'  {fault_type:4} {diam:.3f}"  | {fs:6.1f} | {ss:6.1f} | '
              f'{rs:6.1f} | {rs-fs:+9.1f} dB  (n={n})')
        results.append(dict(label=label, fault=fault_type, diam=diam,
                            fft=fs, stft=ss, rft=rs, delta=rs-fs, n=n))

    # ── False alarm rate on normal bearing ─────────────────────────────────
    print()
    print('=' * 70)
    print('FALSE ALARM RATE — Normal bearing recordings (files 97–100)')
    print('=' * 70)
    threshold_db = 20.0
    counts = {'fft': 0, 'stft': 0, 'rft': 0}
    total  = 0
    if 'normal' in dataset:
        for sig in dataset['normal']:
            for mult in [BPFO_MULT, BPFI_MULT, BSF_MULT]:
                f_t = mult * F_SHAFT
                if fft_snr(sig,  f_t) > threshold_db: counts['fft']  += 1
                if stft_snr(sig, f_t) > threshold_db: counts['stft'] += 1
                if rft_snr(sig,  f_t) > threshold_db: counts['rft']  += 1
                total += 1
        print(f"Threshold : {threshold_db} dB")
        print(f"Files     : {len(dataset['normal'])} normal recordings "
              f"x 3 fault frequencies = {total} checks")
        print()
        print(f"{'Method':6} | {'False Alarms':>14} | {'FAR':>8}")
        print('-' * 36)
        for m in ['fft', 'stft', 'rft']:
            far = counts[m] / max(total, 1) * 100
            print(f"  {m.upper():5} | {counts[m]:>14d} | {far:7.1f}%")
    else:
        print('  [normal bearing files not found]')
        print('  Download 97.mat 98.mat 99.mat 100.mat from:')
        print('  https://engineering.case.edu/bearingdatacenter/download-data-file')
        print('  Place them in the same folder as the fault .mat files and re-run.')

    print()
    print('=' * 70)
    print('REAL DATA BENCHMARK COMPLETE')
    if results:
        rft_wins  = sum(1 for r in results if r['delta'] > 0)
        mean_delta = np.mean([r['delta'] for r in results])
        print(f'RFT vs FFT: {rft_wins}/{len(results)} conditions won | '
              f'mean delta = {mean_delta:+.2f} dB')
    print('=' * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print()
        print('CWRU Real Data Benchmark')
        print('========================')
        print('Usage (bash):')
        print('  python algorithms/rft/applications/vibration/cwru_benchmark_real.py \\')
        print('    algorithms/rft/applications/vibration/')
        print()
        print('Usage (PowerShell):')
        print('  python algorithms/rft/applications/vibration/cwru_benchmark_real.py `')
        print('    algorithms/rft/applications/vibration/')
        print()
        print('Required files in that folder:')
        print('  Fault    : 105 106 118 119 130-133 169-170 185-186')
        print('             197-200 209-210 222-223 234-235  (.mat)')
        print('  Normal   : 97 98 99 100  (.mat)  <- for FAR test')
        print()
        print('Download from:')
        print('  https://engineering.case.edu/bearingdatacenter/download-data-file')
        print()
        print('For synthetic benchmark (no download needed):')
        print('  python algorithms/rft/applications/vibration/cwru_benchmark.py')
        print()
        sys.exit(0)

    run_real_benchmark(sys.argv[1])

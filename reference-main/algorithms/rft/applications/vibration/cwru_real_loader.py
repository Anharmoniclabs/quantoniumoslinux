# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
"""
cwru_real_loader.py
====================
Real CWRU bearing dataset loader for .mat files.

Dataset: Case Western Reserve University Bearing Data Center
URL:     https://engineering.case.edu/bearingdatacenter/download-data-file
Bearing: SKF6205, Drive End (DE) channel, 12 kHz sample rate

CWRU files are downloaded as plain numeric IDs (e.g. 130.mat).
This loader maps those IDs to fault labels automatically.

Numeric ID map (12k Drive End):
  OR 0.007" @6:00  : 130, 131, 132, 133  (1797/1772/1750/1730 rpm)
  OR 0.014" @6:00  : 197, 198, 199, 200
  OR 0.021" @6:00  : 234, 235, 236, 237
  IR 0.007"        : 105, 106, 107, 108
  IR 0.014"        : 169, 170, 171, 172
  IR 0.021"        : 209, 210, 211, 212
  B  0.007"        : 118, 119, 120, 121
  B  0.014"        : 185, 186, 187, 188
  B  0.021"        : 222, 223, 224, 225
  Normal           : 97, 98, 99, 100

Author: Luis M. Minier
Patent: USPTO 19/169,399
Date:   March 2026
"""

from pathlib import Path
import numpy as np

FS_CWRU = 12000.0

# CWRU numeric file ID -> fault label mapping (12k Drive End)
# Keys are the stem of the downloaded filename (e.g. '130' for 130.mat)
_NUMERIC_ID_MAP = {
    # OR 0.007" @6:00
    '130': 'OR_0007', '131': 'OR_0007', '132': 'OR_0007', '133': 'OR_0007',
    # OR 0.014" @6:00
    '197': 'OR_0014', '198': 'OR_0014', '199': 'OR_0014', '200': 'OR_0014',
    # OR 0.021" @6:00
    '234': 'OR_0021', '235': 'OR_0021', '236': 'OR_0021', '237': 'OR_0021',
    # IR 0.007"
    '105': 'IR_0007', '106': 'IR_0007', '107': 'IR_0007', '108': 'IR_0007',
    # IR 0.014"
    '169': 'IR_0014', '170': 'IR_0014', '171': 'IR_0014', '172': 'IR_0014',
    # IR 0.021"
    '209': 'IR_0021', '210': 'IR_0021', '211': 'IR_0021', '212': 'IR_0021',
    # Ball 0.007"
    '118': 'B_0007',  '119': 'B_0007',  '120': 'B_0007',  '121': 'B_0007',
    # Ball 0.014"
    '185': 'B_0014',  '186': 'B_0014',  '187': 'B_0014',  '188': 'B_0014',
    # Ball 0.021"
    '222': 'B_0021',  '223': 'B_0021',  '224': 'B_0021',  '225': 'B_0021',
    # Normal baseline
    '97':  'normal',  '98':  'normal',  '99':  'normal',  '100': 'normal',
}

# Name-pattern fallback (for renamed files)
_NAME_PATTERNS = {
    'OR_0007': ['0.007_OR', '007_OR', 'or007'],
    'OR_0014': ['0.014_OR', '014_OR', 'or014'],
    'OR_0021': ['0.021_OR', '021_OR', 'or021'],
    'IR_0007': ['0.007_IR', '007_IR', 'ir007'],
    'IR_0014': ['0.014_IR', '014_IR', 'ir014'],
    'IR_0021': ['0.021_IR', '021_IR', 'ir021'],
    'B_0007':  ['0.007_b',  '007_b',  'b007'],
    'B_0014':  ['0.014_b',  '014_b',  'b014'],
    'B_0021':  ['0.021_b',  '021_b',  'b021'],
    'normal':  ['normal'],
}

# All known Drive End column names across CWRU .mat files
_DE_KEYS = [
    'DE_time',
    'X097_DE_time', 'X098_DE_time', 'X099_DE_time', 'X100_DE_time',
    'X105_DE_time', 'X106_DE_time', 'X107_DE_time', 'X108_DE_time',
    'X118_DE_time', 'X119_DE_time', 'X120_DE_time', 'X121_DE_time',
    'X130_DE_time', 'X131_DE_time', 'X132_DE_time', 'X133_DE_time',
    'X169_DE_time', 'X170_DE_time', 'X171_DE_time', 'X172_DE_time',
    'X185_DE_time', 'X186_DE_time', 'X187_DE_time', 'X188_DE_time',
    'X197_DE_time', 'X198_DE_time', 'X199_DE_time', 'X200_DE_time',
    'X209_DE_time', 'X210_DE_time', 'X211_DE_time', 'X212_DE_time',
    'X222_DE_time', 'X223_DE_time', 'X224_DE_time', 'X225_DE_time',
    'X234_DE_time', 'X235_DE_time', 'X236_DE_time', 'X237_DE_time',
]


def _load_mat(path: Path) -> np.ndarray:
    """Load Drive End channel from a CWRU .mat file."""
    try:
        from scipy.io import loadmat
        mat = loadmat(str(path))
        # Try known DE keys first
        for key in _DE_KEYS:
            if key in mat:
                return mat[key].squeeze().astype(np.float64)
        # Fallback: first long 1-D numeric array
        for key, val in mat.items():
            if not key.startswith('_') and hasattr(val, 'squeeze'):
                arr = val.squeeze()
                if arr.ndim == 1 and len(arr) > 1000:
                    return arr.astype(np.float64)
        raise KeyError(
            f"No DE channel found in {path.name}. "
            f"Keys: {[k for k in mat if not k.startswith('_')]}"
        )
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                'h5py required for v7.3 .mat files. '
                'Install with: pip install h5py'
            )
        with h5py.File(str(path), 'r') as f:
            for key in _DE_KEYS:
                if key in f:
                    return np.array(f[key]).squeeze().astype(np.float64)
            for key in f.keys():
                arr = np.array(f[key]).squeeze()
                if arr.ndim == 1 and len(arr) > 1000:
                    return arr.astype(np.float64)
        raise KeyError(f'No DE channel found in HDF5 file {path.name}')


def load_cwru_file(mat_path: str, duration_s: float = 2.0,
                  offset_s: float = 0.5) -> np.ndarray:
    """Load one CWRU .mat file, return normalised segment."""
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    raw   = _load_mat(path)
    start = int(offset_s * FS_CWRU)
    end   = start + int(duration_s * FS_CWRU)
    if end > len(raw):
        end   = len(raw)
        start = max(0, end - int(duration_s * FS_CWRU))
    seg = raw[start:end]
    seg = seg - seg.mean()
    peak = np.abs(seg).max()
    if peak > 1e-10:
        seg = seg / peak
    return seg


def discover_cwru_files(data_dir: str) -> dict:
    """
    Scan a directory for CWRU .mat files.
    Matches by numeric ID first (e.g. 130.mat), then by name pattern.
    Returns dict: {label: [Path, ...]}
    """
    folder    = Path(data_dir)
    # Also scan one level of subdirectories (Ball (B)/, Inner Race (IR)/, etc.)
    mat_files = list(folder.glob('*.mat')) + list(folder.glob('*/*.mat'))
    mat_files  = [f for f in mat_files if '.part' not in f.name]

    found = {}

    for f in mat_files:
        stem  = f.stem  # e.g. '130'
        label = None

        # 1. Numeric ID match
        if stem in _NUMERIC_ID_MAP:
            label = _NUMERIC_ID_MAP[stem]

        # 2. Name pattern match (fallback)
        if label is None:
            stem_lower = stem.lower()
            for lbl, patterns in _NAME_PATTERNS.items():
                if any(p in stem_lower for p in patterns):
                    label = lbl
                    break

        if label:
            found.setdefault(label, []).append(f)

    # Sort each list for deterministic ordering
    for label in found:
        found[label] = sorted(found[label])

    return found


def load_cwru_dataset(data_dir: str, duration_s: float = 2.0,
                      verbose: bool = True) -> dict:
    """Load all discovered CWRU .mat files. Returns {label: [signal, ...]}"""
    found = discover_cwru_files(data_dir)
    if not found:
        raise FileNotFoundError(
            f"No CWRU .mat files found in '{data_dir}'.\n"
            f"Download from: https://engineering.case.edu/bearingdatacenter/download-data-file\n"
            f"Files should be named like: 130.mat, 105.mat, 118.mat (CWRU numeric IDs)\n"
            f"Or: 0.007_OR_6205_1797.mat (descriptive names)"
        )
    dataset = {}
    for label, paths in found.items():
        signals = []
        for p in paths:
            try:
                sig = load_cwru_file(str(p), duration_s=duration_s)
                signals.append(sig)
                if verbose:
                    print(f'  [OK] {label:12} <- {p.name}  ({len(sig)} samples)')
            except Exception as e:
                if verbose:
                    print(f'  [!!] {label:12} <- {p.name}  ERROR: {e}')
        if signals:
            dataset[label] = signals
    if verbose:
        total = sum(len(v) for v in dataset.values())
        print(f'\nLoaded {total} signal(s) across {len(dataset)} fault classes.')
    return dataset


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python cwru_real_loader.py <path_to_cwru_folder>')
        sys.exit(0)
    data_dir = sys.argv[1]
    print(f"Scanning '{data_dir}'...\n")
    dataset = load_cwru_dataset(data_dir, verbose=True)
    print('\nSummary:')
    for label, sigs in dataset.items():
        print(f'  {label:12}: {len(sigs)} file(s)')

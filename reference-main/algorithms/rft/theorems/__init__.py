# algorithms/rft/theorems/__init__.py
# =====================================
# Theoretical results for the Resonant Fourier Transform.
#
# STATUS: Theory files are physically still in algorithms/rft/core/
# pending full import-safe migration. This package re-exports them
# from their current location so downstream code can begin importing
# from algorithms.rft.theorems immediately.
#
# FILES RE-EXPORTED FROM core/ (to be physically moved in next pass):
#   transform_theorems.py            — Theorems 6, 8, 15, Theorem E
#   golden_uncertainty_principle.py  — φ-basis Heisenberg bound
#   maassen_uffink_uncertainty.py    — Maassen-Uffink entropic bounds
#   sharp_coherence_bounds.py        — Coherence bounds RFT vs DFT
#   theorem8_bootstrap_verification  — Bootstrap verification Theorem 8
#
# Use:
#   from algorithms.rft.theorems import transform_theorems

try:
    from algorithms.rft.core import transform_theorems  # noqa: F401
except ImportError:
    pass

try:
    from algorithms.rft.core import golden_uncertainty_principle  # noqa: F401
except ImportError:
    pass

try:
    from algorithms.rft.core import maassen_uffink_uncertainty  # noqa: F401
except ImportError:
    pass

try:
    from algorithms.rft.core import sharp_coherence_bounds  # noqa: F401
except ImportError:
    pass

# algorithms/rft/legacy/__init__.py
# ===================================
# Legacy and archived RFT implementations.
#
# FILES IN THIS DIRECTORY:
#
#   rft_phi_legacy.py
#     Pre-Gram-normalization implementation (locked).
#     Preserved for benchmark reproducibility only.
#     DO NOT use in new code.
#     Canonical replacement: algorithms.rft.core.resonant_fourier_transform
#
#   dft_distance_certificate.py
#     Certified lower-bound distance from RFT to DFT under diagonal-unitary
#     and permutation symmetries. PREVIOUSLY NAMED: absolute_novelty.py.
#     Renamed because the original name was not appropriate for production code.
#     The mathematics is valid and citable (see module docstring).
#     Old import names preserved as aliases for backward compatibility.

from .dft_distance_certificate import (
    certified_dft_distance_lower_bound,
    certified_abs_novelty_lower_bound_to_dft,  # backward compat alias
    heuristic_dft_distance_upper_bound,
    heuristic_abs_novelty_upper_bound,          # backward compat alias
    best_diagonal_phase_alignment,
    best_diagonal_phase_alignment_for_fixed_row_perm,  # backward compat alias
    AlignmentResult,
)

__all__ = [
    "certified_dft_distance_lower_bound",
    "certified_abs_novelty_lower_bound_to_dft",
    "heuristic_dft_distance_upper_bound",
    "heuristic_abs_novelty_upper_bound",
    "best_diagonal_phase_alignment",
    "best_diagonal_phase_alignment_for_fixed_row_perm",
    "AlignmentResult",
]

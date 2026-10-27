#!/usr/bin/env python3
"""Compare Toeplitz-FFT and Chebyshev inverse-square-root actions to Lanczos."""

from __future__ import annotations

import numpy as np


PHI = (1.0 + np.sqrt(5.0)) / 2.0


def phi_fft(vector: np.ndarray, adjoint: bool) -> np.ndarray:
    """Apply raw Phi or Phi.H using Bluestein convolution in O(N log N)."""
    n = vector.size
    fft_size = 1 << (3 * n - 2).bit_length()
    indices = np.arange(n, dtype=np.float64)
    source = np.zeros(fft_size, dtype=np.complex128)
    kernel = np.zeros(fft_size, dtype=np.complex128)

    if adjoint:
        offsets = np.arange(1 - n, n + 1, dtype=np.float64)
        source[:n] = vector * np.exp(-1j * np.pi * PHI * indices**2)
        kernel[(offsets + n - 1).astype(np.intp)] = np.exp(
            1j * np.pi * PHI * offsets**2
        )
        output = np.fft.ifft(np.fft.fft(source) * np.fft.fft(kernel))
        result = output[n:n + n] * np.exp(
            -1j * np.pi * PHI * (indices + 1)**2
        )
    else:
        offsets = np.arange(-n, n - 1, dtype=np.float64)
        source[:n] = vector * np.exp(
            1j * np.pi * PHI * (indices + 1)**2
        )
        kernel[(offsets + n).astype(np.intp)] = np.exp(
            -1j * np.pi * PHI * offsets**2
        )
        output = np.fft.ifft(np.fft.fft(source) * np.fft.fft(kernel))
        result = output[n - 1:n - 1 + n] * np.exp(1j * np.pi * PHI * indices**2)

    return result / np.sqrt(n)


def build_dense_gram(n: int) -> np.ndarray:
    samples = np.arange(n, dtype=np.float64)
    frequencies = np.mod(np.arange(1, n + 1, dtype=np.float64) * PHI, 1.0)
    phi = np.exp(2j * np.pi * np.outer(samples, frequencies)) / np.sqrt(n)
    return phi.conj().T @ phi


def build_toeplitz_fft_matvec(n: int):
    offsets = np.arange(n, dtype=np.float64)
    phase = np.pi * offsets * PHI
    denominator = n * np.sin(phase)
    row = np.ones(n, dtype=np.complex128)
    row[1:] = (
        np.exp(1j * (n - 1) * phase[1:])
        * np.sin(n * phase[1:])
        / denominator[1:]
    )
    embedding = np.concatenate((row.conj(), np.zeros(1), row[:0:-1]))
    embedding_fft = np.fft.fft(embedding)

    def matvec(vector: np.ndarray) -> np.ndarray:
        padded = np.zeros(2 * n, dtype=np.complex128)
        padded[:n] = vector
        return np.fft.ifft(embedding_fft * np.fft.fft(padded))[:n]

    return matvec


def lanczos_inverse_sqrt(matvec, vector: np.ndarray, steps: int) -> np.ndarray:
    """Reference full-reorthogonalization Lanczos action used for comparison."""
    n = vector.size
    steps = min(steps, n)
    norm = np.linalg.norm(vector)
    basis = np.zeros((n, steps), dtype=np.complex128)
    basis[:, 0] = vector / norm
    diagonal = np.zeros(steps, dtype=np.float64)
    off_diagonal = np.zeros(max(0, steps - 1), dtype=np.float64)
    actual_steps = steps

    for j in range(steps):
        residual = matvec(basis[:, j])
        if j:
            residual -= off_diagonal[j - 1] * basis[:, j - 1]
        diagonal[j] = np.vdot(basis[:, j], residual).real
        residual -= diagonal[j] * basis[:, j]
        for i in range(j + 1):
            residual -= np.vdot(basis[:, i], residual) * basis[:, i]
        beta = np.linalg.norm(residual)
        if j == steps - 1 or beta <= 1e-14 * norm:
            actual_steps = j + 1
            break
        off_diagonal[j] = beta
        basis[:, j + 1] = residual / beta

    tridiagonal = np.diag(diagonal[:actual_steps])
    if actual_steps > 1:
        off = off_diagonal[:actual_steps - 1]
        tridiagonal += np.diag(off, 1) + np.diag(off, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    if eigenvalues[0] <= 0.0:
        raise ArithmeticError("Lanczos projected Gram matrix is not positive definite")
    first_row = eigenvectors[0, :]
    weights = (first_row / np.sqrt(eigenvalues)) @ eigenvectors.T
    return norm * (basis[:, :actual_steps] @ weights)


def chebyshev_inverse_sqrt(matvec, vector: np.ndarray, lower: float, upper: float,
                           degree: int) -> np.ndarray:
    """Apply a Chebyshev interpolant on the supplied spectral interval."""
    count = degree + 1
    nodes = np.cos(np.pi * (np.arange(count) + 0.5) / count)
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)
    values = 1.0 / np.sqrt(center + radius * nodes)
    coefficients = np.polynomial.chebyshev.chebfit(nodes, values, degree)

    def scaled_matvec(value: np.ndarray) -> np.ndarray:
        return (matvec(value) - center * value) / radius

    previous = vector
    result = coefficients[0] * previous
    if degree == 0:
        return result
    current = scaled_matvec(vector)
    result = result + coefficients[1] * current
    for order in range(2, degree + 1):
        following = 2.0 * scaled_matvec(current) - previous
        result = result + coefficients[order] * following
        previous, current = current, following
    return result


def inverse_sqrt_rational_poles(lower: float, upper: float, count: int):
    """Gauss-Legendre discretization of G^-1/2 = 2/pi int (G+t^2 I)^-1 dt."""
    scale = np.sqrt(lower * upper)
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(count)
    angles = (nodes + 1.0) * (np.pi / 4.0)
    poles = scale**2 * np.tan(angles)**2
    weights = scale * quadrature_weights / (2.0 * np.cos(angles)**2)
    return poles, weights


def multishift_cg(matvec, vector: np.ndarray, shifts: np.ndarray,
                  tolerance: float = 1e-11, max_iterations: int | None = None,
                  include_base: bool = True):
    """Solve all (G + shift I)x=vector systems with shared CG and early freezing."""
    n = vector.size
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 1 or shifts.size == 0 or np.any(shifts < 0.0):
        raise ValueError("shifts must be a non-empty vector of non-negative reals")
    base = float(shifts.min()) if include_base else 0.0
    deltas = shifts - base
    if np.any(deltas < 0.0):
        raise ValueError("all relative shifts must be non-negative")

    limit = min(n, max_iterations if max_iterations is not None else n)
    solutions = np.zeros((shifts.size, n), dtype=np.complex128)
    shifted_directions = np.tile(vector, (shifts.size, 1)).astype(np.complex128)
    z_previous = np.ones(shifts.size, dtype=np.float64)
    z_current = np.ones(shifts.size, dtype=np.float64)
    active = np.ones(shifts.size, dtype=bool)
    residual = np.asarray(vector, dtype=np.complex128).copy()
    direction = residual.copy()
    residual_norm_sq = float(np.vdot(residual, residual).real)
    vector_norm = np.sqrt(residual_norm_sq)
    alpha_previous = 1.0
    beta_previous = 0.0

    for iteration in range(limit):
        gram_direction = matvec(direction)
        if base:
            gram_direction += base * direction
        denominator_cg = float(np.vdot(direction, gram_direction).real)
        if denominator_cg <= 0.0 or not np.isfinite(denominator_cg):
            raise ArithmeticError("base Gram operator is not positive definite")
        alpha = residual_norm_sq / denominator_cg
        residual_next = residual - alpha * gram_direction
        residual_next_norm_sq = float(np.vdot(residual_next, residual_next).real)
        beta = residual_next_norm_sq / residual_norm_sq

        denominator_z = (
            alpha * beta_previous * (z_previous - z_current)
            + z_previous * alpha_previous * (1.0 + deltas * alpha)
        )
        z_next = z_current.copy()
        z_next[active] = (
            z_current[active] * z_previous[active] * alpha_previous
            / denominator_z[active]
        )
        if not np.all(np.isfinite(z_next[active])):
            raise FloatingPointError("non-finite multi-shift recurrence")

        ratio = np.zeros(shifts.size, dtype=np.float64)
        ratio[active] = z_next[active] / z_current[active]
        if not np.all(np.isfinite(ratio[active])):
            raise FloatingPointError("non-finite multi-shift direction ratio")
        solutions[active] += (
            alpha * ratio[active, None] * shifted_directions[active]
        )
        shifted_directions[active] = (
            z_next[active, None] * residual_next
            + beta * ratio[active, None]**2 * shifted_directions[active]
        )

        estimated_shifted_residual = np.abs(z_next) * np.sqrt(residual_next_norm_sq)
        converged = active & (estimated_shifted_residual <= tolerance * vector_norm)
        active[converged] = False
        if not np.any(active):
            return solutions, iteration + 1, "all shifts converged"
        if np.sqrt(residual_next_norm_sq) <= tolerance * vector_norm:
            return solutions, iteration + 1, "base converged before shifts"

        residual = residual_next
        direction = residual_next + beta * direction
        residual_norm_sq = residual_next_norm_sq
        z_previous[active], z_current[active] = (
            z_current[active], z_next[active]
        )
        alpha_previous = alpha
        beta_previous = beta

    return solutions, limit, "iteration limit"


def relative_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(approximation - reference) / np.linalg.norm(reference))


def main() -> None:
    sizes = (32, 64, 128, 256, 512)
    degrees = (8, 16, 32, 64, 128)
    lanczos_steps = (8, 16, 32, 64, 128)
    rational_terms = 40
    tolerance = 1e-6
    rng = np.random.default_rng(20261027)

    print("N  cond(G)       Toeplitz matvec rel.err  Cheb degree/error (oracle bounds)    Lanczos steps/error")
    for n in sizes:
        gram = build_dense_gram(n)
        samples = np.arange(n, dtype=np.float64)
        frequencies = np.mod(np.arange(1, n + 1, dtype=np.float64) * PHI, 1.0)
        phi = np.exp(2j * np.pi * np.outer(samples, frequencies)) / np.sqrt(n)
        matvec = build_toeplitz_fft_matvec(n)
        vector = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        vector = vector.astype(np.complex128)
        toeplitz_error = relative_error(matvec(vector), gram @ vector)
        raw_adjoint_error = relative_error(phi_fft(vector, True), phi.conj().T @ vector)
        raw_synthesis_error = relative_error(phi_fft(vector, False), phi @ vector)

        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        if eigenvalues[0] <= 0.0:
            print(f"{n:3d}  nonpositive computed lambda_min={eigenvalues[0]:.3e}; skip spectral actions")
            continue
        inverse_sqrt_vector = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ (
            eigenvectors.conj().T @ vector
        )
        condition = eigenvalues[-1] / eigenvalues[0]

        cheb_results = []
        for degree in degrees:
            estimate = chebyshev_inverse_sqrt(
                matvec, vector, eigenvalues[0], eigenvalues[-1], degree
            )
            error = relative_error(estimate, inverse_sqrt_vector)
            cheb_results.append((degree, error))
        cheb_hit = next((degree for degree, error in cheb_results if error <= tolerance), None)

        lanczos_results = []
        for steps in lanczos_steps:
            estimate = lanczos_inverse_sqrt(matvec, vector, steps)
            error = relative_error(estimate, inverse_sqrt_vector)
            lanczos_results.append((steps, error))
        lanczos_hit = next((steps for steps, error in lanczos_results if error <= tolerance), None)

        degree = 512
        coefficients = chebyshev_inverse_sqrt(
            matvec, phi_fft(vector, True), eigenvalues[0], eigenvalues[-1], degree
        )
        reconstructed = phi_fft(
            chebyshev_inverse_sqrt(
                matvec, coefficients, eigenvalues[0], eigenvalues[-1], degree
            ),
            False,
        )
        roundtrip_error = relative_error(reconstructed, vector)

        poles, rational_weights = inverse_sqrt_rational_poles(
            eigenvalues[0], eigenvalues[-1], rational_terms
        )
        shifted_solutions, rational_iterations, rational_status = multishift_cg(
            matvec, vector, poles, tolerance=1e-11, max_iterations=n
        )
        rational_action = rational_weights @ shifted_solutions
        rational_error = relative_error(rational_action, inverse_sqrt_vector)
        unshifted_solutions, _, unshifted_status = multishift_cg(
            matvec, vector, poles, tolerance=1e-11, max_iterations=n,
            include_base=False,
        )
        base_mode_error = relative_error(
            rational_weights @ unshifted_solutions, rational_action
        )
        rational_forward = rational_weights @ multishift_cg(
            matvec, phi_fft(vector, True), poles, tolerance=1e-11,
            max_iterations=n,
        )[0]
        rational_roundtrip = phi_fft(
            rational_weights @ multishift_cg(
                matvec, rational_forward, poles, tolerance=1e-11,
                max_iterations=n,
            )[0],
            False,
        )
        rational_roundtrip_error = relative_error(rational_roundtrip, vector)
        spectral_grid = np.geomspace(eigenvalues[0], eigenvalues[-1], 5000)
        rational_scalar_error = float(np.max(np.abs(
            (rational_weights[:, None] / (spectral_grid[None, :] + poles[:, None])).sum(axis=0)
            * np.sqrt(spectral_grid) - 1.0
        )))
        if max(toeplitz_error, raw_adjoint_error, raw_synthesis_error) > 1e-8:
            raise AssertionError(f"FFT operator mismatch at N={n}")
        if rational_status != "all shifts converged" or unshifted_status != "all shifts converged":
            raise AssertionError(f"multi-shift CG did not converge at N={n}")
        if rational_error > 1e-6 or rational_roundtrip_error > 1e-6:
            raise AssertionError(f"rational RFT accuracy target missed at N={n}")
        if base_mode_error > 1e-10:
            raise AssertionError(f"base-shift modes disagree at N={n}")

        cheb_text = ", ".join(f"{degree}:{error:.1e}" for degree, error in cheb_results)
        lanczos_text = ", ".join(f"{steps}:{error:.1e}" for steps, error in lanczos_results)
        print(f"{n:3d}  {condition:11.3e}  {toeplitz_error:11.3e}             {cheb_text}")
        print(f"     first Cheb degree <= {tolerance:.0e}: {cheb_hit}; first Lanczos steps: {lanczos_hit}")
        print(f"     Lanczos steps:error: {lanczos_text}")
        print(
            f"     raw Phi/Phi.H rel.err: {raw_synthesis_error:.2e}/"
            f"{raw_adjoint_error:.2e}; degree-{degree} roundtrip: {roundtrip_error:.2e}"
        )
        print(
            f"     rational terms/CG iterations: {rational_terms}/{rational_iterations} "
            f"({rational_status}); scalar/action rel.err: "
            f"{rational_scalar_error:.2e}/{rational_error:.2e}"
        )
        print(
            f"     include_base mode agreement: {base_mode_error:.2e} "
            f"({unshifted_status}); rational roundtrip: {rational_roundtrip_error:.2e}"
        )


if __name__ == "__main__":
    main()
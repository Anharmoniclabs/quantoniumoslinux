#!/usr/bin/env python3
"""Small structural checks for ARITHMETIC_ROUTES.md; not an RH experiment.

Integer/rational checks are exact. The q=8 exhaustive random-sign ensemble
uses floating-point linear algebra and an explicit tolerance. No growth scan,
new dependencies, output files, or changes to existing probe checkpoints.
"""
from fractions import Fraction
from itertools import product
from math import gcd, isqrt

import numpy as np


def factor(n):
    """Independent trial division, sufficient for these small fixed checks."""
    factors = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def mobius(n):
    factors = factor(n)
    return 0 if any(e > 1 for e in factors.values()) else (-1) ** len(factors)


def divisors(n):
    result = []
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            result.append(d)
            if d * d != n:
                result.append(n // d)
    return result


def modified_sign(n, primes):
    factors = factor(n)
    if any(e > 1 for e in factors.values()):
        return 0
    return (-1) ** sum(p in primes for p in factors)


def convolution_factor(n, primes):
    result = 1
    for p, exponent in factor(n).items():
        if p not in primes:
            return 0
        result *= 2 * (-1) ** exponent
    return result


def check_arithmetic():
    limit = 512
    mu = [0] + [mobius(n) for n in range(1, limit + 1)]
    for n in range(1, limit + 1):
        assert sum(mu[d] for d in divisors(n)) == int(n == 1)
    for p in (2, 3, 5, 7):
        total = 0
        for n in range(1, limit + 1):
            total += mu[n]
            shell = sum(mu[j] for j in range(n // p + 1, n + 1) if j % p)
            assert total == shell, (p, n)
    print("EXACT: divisor identity and prime pairing, cutoffs 1..512: PASS")

    for primes in (frozenset((2,)), frozenset((2, 3, 5, 7))):
        g = [0] + [modified_sign(n, primes) for n in range(1, limit + 1)]
        assert all(g[n] ** 2 == mu[n] ** 2 for n in range(1, limit + 1))
        for n in range(1, limit + 1):
            convolution = sum(mu[n // d] ** 2 * convolution_factor(d, primes)
                              for d in divisors(n))
            assert g[n] == convolution, (primes, n)
        for a in range(1, 33):
            for b in range(1, min(32, limit // a) + 1):
                if gcd(a, b) == 1:
                    assert g[a * b] == g[a] * g[b]
        for p in primes:
            total = 0
            for n in range(1, limit + 1):
                total += g[n]
                assert total == sum(g[j] for j in range(n // p + 1, n + 1)
                                    if j % p)
    g = [0] + [modified_sign(n, {2, 3, 5, 7}) for n in range(1, 12)]
    assert g[1:11] == mu[1:11]
    assert g[11] == 1 and mu[11] == -1
    print("EXACT: signed multiplicative controls, support and convolution: PASS")
    print("EXACT: control with primes {2,3,5,7} agrees through 10; differs at 11")


def fractional_part(x):
    return x - x.numerator // x.denominator


def cell_error(coefficients, j):
    return 1 - sum(a * Fraction(j % n, n) for n, a in coefficients.items())


def finite_fit(limit):
    coefficients = {n: Fraction(-mobius(n)) for n in range(2, limit + 1)}
    h = sum(Fraction(mobius(n), n) for n in range(1, limit + 1))
    coefficients[limit + 1] = (limit + 1) * h
    return coefficients


def error_certificate(coefficients, cells):
    """Exact rational lower/upper bounds for the full infinite squared norm."""
    partial = sum(cell_error(coefficients, j) ** 2 / (j * (j + 1))
                  for j in range(1, cells + 1))
    envelope = 1 + sum(abs(a) for a in coefficients.values())
    return partial, partial + envelope ** 2 / (cells + 1)


def check_spatial_criterion():
    for n in range(2, 25):
        for j in range(1, 65):
            x = (Fraction(1, j) + Fraction(1, j + 1)) / 2
            direct = fractional_part(1 / (n * x)) - fractional_part(1 / x) / n
            assert direct == Fraction(j % n, n)
    for limit in (2, 5, 12, 24):
        coefficients = finite_fit(limit)
        assert all(cell_error(coefficients, j) == 0 for j in range(1, limit + 1))
        lower, upper = error_certificate(coefficients, limit)
        more_lower, more_upper = error_certificate(coefficients, 4 * limit)
        assert lower == 0
        assert lower <= more_lower <= more_upper <= upper
    coefficients = finite_fit(5)
    assert coefficients == {2: Fraction(1), 3: Fraction(1), 4: Fraction(0),
                            5: Fraction(1), 6: Fraction(-1, 5)}
    assert cell_error(coefficients, 6) == Fraction(4, 5)
    assert error_certificate(coefficients, 6)[0] == Fraction(8, 525)
    print("EXACT: fractional-part cell identity and finite fits: PASS")
    print("EXACT: L=5 sampled squared error = 0; full squared error >= 8/525")
    print("EXACT: rational tail certificates shrink when more cells are included: PASS")


def check_random_control():
    q = 8
    alpha = (np.sqrt(5.0) - 1) / 2
    raw = np.exp(2j * np.pi * np.outer(np.arange(q), np.arange(1, q + 1) * alpha))
    raw /= np.sqrt(q)
    inverse = np.linalg.inv(raw)
    mu = np.array([mobius(n) for n in range(1, q + 1)], dtype=float)
    support = np.flatnonzero(mu)
    signals = np.zeros((2 ** len(support), q))
    signals[:, support] = np.array(list(product((-1, 1), repeat=len(support))))
    coefficients = signals @ inverse.T
    diagonal = np.abs(inverse) ** 2 @ mu ** 2
    mean_square = np.mean(np.abs(coefficients) ** 2, axis=0)
    error = float(np.max(np.abs(mean_square - diagonal)))
    assert np.allclose(mean_square, diagonal, rtol=0, atol=2e-12)
    assert np.allclose(np.mean(coefficients, axis=0), 0, rtol=0, atol=2e-12)
    expected = np.zeros(q, dtype=complex)
    expected[0] = np.sqrt(q)
    assert np.allclose(inverse @ (np.sqrt(q) * raw[:, 0]), expected,
                       rtol=0, atol=2e-12)
    print(f"NUMERICAL: all {len(signals)} independent-sign controls at q=8: PASS")
    print(f"NUMERICAL: mean |c_k|^2 = D_k; largest rounding difference {error:.2e}")


def main():
    check_arithmetic()
    check_spatial_criterion()
    check_random_control()
    print("Structural checks complete. All-size statements require the proofs in")
    print("ARITHMETIC_ROUTES.md. RH and the Mobius coefficient hypothesis remain unproved.")


if __name__ == "__main__":
    main()

"""Independent checks for the finite experiment; not number-theoretic proofs."""
import unittest

from rft_fibonacci_prefix import fibonacci_frames, mobius_sieve, prefix_frame
import numpy as np


def trial_mu(n):
    sign = 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            sign = -sign
            if n % p == 0:
                return 0
        p += 1
    return -sign if n > 1 else sign


class PrefixExperimentTests(unittest.TestCase):
    def test_sieve_against_trial_factorization(self):
        actual = mobius_sieve(2584)
        expected = [0] + [trial_mu(n) for n in range(1, 2585)]
        np.testing.assert_array_equal(actual, expected)

    def test_all_cutoffs_and_non_fibonacci_limit(self):
        for limit in (1, 4, 13, 20):
            frames = fibonacci_frames(limit)
            mu = mobius_sieve(frames[-1])
            observed = []
            previous = 0
            for frame in frames:
                last = min(frame, limit)
                ms, exact, _, _, _ = prefix_frame(frame, mu, previous + 1, last)
                observed.extend(ms.tolist())
                np.testing.assert_array_equal(exact, np.cumsum(mu)[ms])
                previous = last
            self.assertEqual(observed, list(range(1, limit + 1)))

    def test_prefix_bound_against_gram_inverse_sqrt(self):
        frame = 13
        mu = mobius_sieve(frame)
        ms, exact, bounds, _, _ = prefix_frame(frame, mu, 6, 11)
        phi = (np.longdouble(1) + np.sqrt(np.longdouble(5))) / 2
        freqs = np.asarray(np.remainder(
            (np.arange(frame, dtype=np.longdouble) + 1) * phi, 1
        ), dtype=float)
        raw = np.exp(2j * np.pi * np.outer(np.arange(frame), freqs)) / np.sqrt(frame)
        eigenvalues, vectors = np.linalg.eigh(raw.conj().T @ raw)
        U = raw @ ((vectors / np.sqrt(eigenvalues)) @ vectors.conj().T)
        b = U.conj().T @ mu[1:]
        for i, m in enumerate(ms):
            mask = np.zeros(frame)
            mask[:m] = 1
            a = U.conj().T @ mask
            self.assertLess(abs(np.vdot(a, b) - exact[i]), 1e-11)
            self.assertAlmostEqual(np.sum(np.abs(a) * np.abs(b)), bounds[i], places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)

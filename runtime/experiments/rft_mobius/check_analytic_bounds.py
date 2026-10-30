"""Finite normalization checks for ANALYTIC_BOUNDS.md; not an RH test."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = os.environ.get("RFT_THREADS", "1")

from decimal import Decimal, localcontext
import numpy as np
from rft_fibonacci_prefix import mobius_sieve


def main():
    with localcontext() as context:
        context.prec = 100
        phi = (Decimal(1) + Decimal(5).sqrt()) / 2
        alpha = 1 / phi
        L0 = 1 / (2 * Decimal(5).sqrt())
        previous, current = 0, 1
        for index in range(1, 101):
            q, p = current, previous
            error = q * alpha - p
            assert abs(error - (-1)**(index-1) * phi**(-index)) < Decimal('1e-70')
            assert 0 <= (q-1) * phi**(-index) / 2 < L0
            if q >= 2:
                assert q * phi**(-index) < Decimal('0.5')
            previous, current = current, previous + current
        print('Fibonacci displacement identities through index 100: PASS')
        print('q   centered displacement   ||T-Q||_2   theoretical upper')
        for q, p in [(2,1),(3,2),(5,3),(8,5),(13,8),(34,21),(55,34),(89,55),(144,89),(233,144)]:
            delta = alpha - Decimal(p)/q
            ks = np.arange(1, q+1)
            residues = (ks*p) % q
            shifts = np.array([float(q * (Decimal(int(k)) - Decimal(q+1)/2) * delta) for k in ks])
            ts = 2*np.pi*(np.arange(q)-(q-1)/2)/q
            Q = np.exp(1j*np.outer(ts,residues))/np.sqrt(q)
            T = np.exp(1j*np.outer(ts,residues+shifts))/np.sqrt(q)
            L = float(np.max(np.abs(shifts)))
            rho = 1-np.cos(np.pi*L)+np.sin(np.pi*L)
            norm = float(np.linalg.norm(T-Q,2))
            assert norm <= rho + 1e-12
            freqs = np.array([float((int(k)*alpha) % 1) for k in ks])
            raw = np.exp(2j*np.pi*np.outer(np.arange(q),freqs))/np.sqrt(q)
            sv = np.linalg.svd(raw,compute_uv=False)
            np.testing.assert_allclose(sv,np.linalg.svd(T,compute_uv=False),atol=1e-11,rtol=0)
            assert sv[-1] >= 1-rho-1e-12 and sv[0] <= 1+rho+1e-12
            prefix = np.cumsum(raw,axis=0)
            assert np.max(np.sum(np.abs(prefix),axis=1)) <= np.sqrt(q)*(3+np.log(q))+1e-10
            mu = mobius_sieve(q)[1:]
            coefficients = np.linalg.solve(raw,mu)
            np.testing.assert_allclose(prefix@coefficients,np.cumsum(mu),atol=1e-10,rtol=0)
            print(f'{q:<4} {L:>20.10f} {norm:>12.8f} {rho:>19.8f}')
    print('Even/odd centering, singular bounds, prefix l1 bound and coefficient identity: PASS')
    print('The accompanying mathematical arguments establish the stated uniform geometry bounds.')
    print('The Mobius coefficient hypothesis remains unproved.')


if __name__ == '__main__':
    main()

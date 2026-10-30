# Arithmetic, probability, and spatial approximation

This audit connects the Fibonacci RFT experiment to exact prime pairing,
signed multiplicative controls, probability, and Hilbert-space approximation.
It gives proofs of supporting statements and explicit tests of tempting
inferences. It does not prove RH or hypothesis (H). No novelty is claimed.
Notation for Phi_q, c_q, and the constant d is in
[ANALYTIC_BOUNDS.md](ANALYTIC_BOUNDS.md).

## 1. Exact cancellation by prime pairing [PROVEN]

For every prime p and integer N>=1,

    M(N) = sum_{N/p < n <= N, p does not divide n} mu(n).          (1)

To prove this, let A_p(x)=sum_{n<=x, p does not divide n}mu(n).
Terms divisible by p exactly once contribute -A_p(N/p), since
mu(pn)=-mu(n) when p does not divide n. Terms divisible by p^2 vanish.
Thus M(N)=A_p(N)-A_p(N/p), which is (1).

For p=2 this leaves only odd integers in (N/2,N]. It is an exact multiscale
pairing mechanism, valid without a cutoff. It is not a square-root estimate:
the triangle inequality still counts order N terms in the surviving shell.
The signs in that shell have not been proved independent or sufficiently
cancelling. Applying the same identity again does not establish independence.

The related divisor identity has an operator formulation. For 1<=i,j<=N,
let D_N[i,j]=1 if j divides i and 0 otherwise. Then D_N mu_N=e_1 exactly,
so M(N)=1^T D_N^(-1)e_1. A change of coordinates preserves this matrix
element; stability of the coordinate transform alone does not bound it.

## 2. A signed multiplicative counterfamily [PROVEN]

Fix a finite set P of primes and define

    g_P(n) = mu(n)^2 (-1)^(number of primes in P dividing n).

This sequence takes values -1,0,+1, is multiplicative on coprime arguments,
and has exactly the squarefree support of mu. If P contains every prime
<=B, then g_P(n)=mu(n) for every n<=B. It obeys the prime-pairing identity
(1), with M replaced by its own partial sum, for every p in P.

Nevertheless,

    sum_{n<=x} g_P(n) = c_P x + O_P(sqrt(x)),
    c_P = (6/pi^2) product_{p in P} (p-1)/(p+1) > 0.             (2)

Here is an elementary proof, including explicit dependence on P.
Squarefree counting gives

    Q(x) = sum_{n<=x}mu(n)^2
         = sum_{d<=sqrt(x)} mu(d) floor(x/d^2),
    |Q(x) - (6/pi^2)x| <= 3 sqrt(x), x>=1.

The floor errors total at most sqrt(x). The omitted absolutely convergent
series contributes at most 2sqrt(x), using floor(sqrt(x))>=sqrt(x)/2.
The constant is sum mu(d)/d^2=1/zeta(2)=6/pi^2, from the absolutely
convergent Euler product.

Define a multiplicative h_P by h_P(1)=1 and, for k>=1,

    h_P(p^k) = 2(-1)^k if p is in P, and 0 otherwise.

Then g_P=mu^2*h_P by Dirichlet convolution: at a selected prime the local
power series multiply as (1+z)(1-z)/(1+z)=1-z. Moreover,

    sum_d h_P(d)/d = product_{p in P}(p-1)/(p+1),
    B_P := sum_d |h_P(d)|/sqrt(d)
         = product_{p in P}(sqrt(p)+1)/(sqrt(p)-1) < infinity.

Convolving Q and separating d<=x from d>x proves the explicit inequality

    |sum_{n<=x}g_P(n) - c_P x| <= 3 B_P sqrt(x).                (3)

For d<=x use the bound on Q(x/d). For d>x, bound the omitted main term
(6/pi^2)x|h_P(d)|/d by (6/pi^2)sqrt(x)|h_P(d)|/sqrt(d).
Since 6/pi^2<3, the two contributions total at most 3B_P sqrt(x).

For example P={2} gives c_P=2/pi^2. These are not counterexamples to RH:
g_P(p)=+1 for primes outside P, whereas mu(p)=-1 for every prime.
They are counterexamples to deriving square-root cancellation from just
signed multiplicativity, squarefree support, any finite initial segment,
and finitely many prime-pairing rules.

The connection to the RFT is quantitative. Because g_P^2=mu^2, every
inverse-frame row has exactly the same diagonal term
D_k=sum_n |w_kn|^2 mu(n)^2 for both sequences. But (3) and the raw-prefix
bound in ANALYTIC_BOUNDS.md imply, for all sufficiently large Fibonacci q,

    ||Phi_q^(-1)(g_P)_q||_infinity
       >= (c_P/2) sqrt(q)/(3+log q).                            (4)

Thus (H) fails for g_P, despite identical geometry and diagonal energy.
A successful Mobius proof must use its prime signs at unbounded scales.
One cannot let P grow with x in (3) while treating c_P and B_P as fixed.

## 3. Why the random control stays small [PROVEN FOR THAT MODEL]

Let xi_n be independent signs with equal probabilities and set
X_n=|mu(n)|xi_n. These are independent signs at the squarefree integers,
as in the probe, not a random multiplicative function generated at primes.
For a row w of Phi_q^(-1), geometry gives sum|w_n|^2<=d^(-2).

For real coefficients a_n, independence and cosh(u)<=exp(u^2/2) give

    E exp(lambda sum a_n xi_n) <= exp(lambda^2 sum a_n^2 / 2).

Chernoff's inequality applied to real and imaginary parts, followed by
a union bound over the q rows, therefore gives

    P(max_k |(Phi_q^(-1)X)_k| >= t)
        <= 4q exp(-d^2 t^2/4).                                (5)

Take t=(4/d)sqrt(log q), q>=2. The probability is at most 4q^(-3).
Summing over Fibonacci sizes and applying the first Borel-Cantelli lemma
proves that this threshold is exceeded only finitely often, almost surely.
Thus the independent-sign model satisfies (H) almost surely. Independence
between different frame sizes is not required for that conclusion.

Also E|c_k|^2=D_k and E R_k=0 exactly: distinct sign products have mean zero.
The checker enumerates all 64 sign assignments at q=8, rather than estimating
that mean with a Monte Carlo sample.

The missing transfer is precise: deterministic Mobius signs do not satisfy
the independence assumption used to factor the exponential moment. A similar
plot, similar RMS, or a bounded Gram condition number does not supply it.
Random signs chosen at primes would also require a different argument,
because their multiplicative extensions at integers are dependent.

## 4. A spatial criterion with exactly RH's scope [REPORTED + DERIVATION]

The Nyman-Beurling criterion, strengthened by Baez-Duarte, states that RH
is equivalent to 1_(0,1) belonging to the L2(0,infinity) closure of the span
of rho_n(x)={1/(nx)}, n>=1. Source: L. Baez-Duarte, Theorem 1.1,
https://arxiv.org/abs/math/0202141 (published in 2003).

Here is a constrained spatial formulation and its arithmetic cells. Set

    psi_n(x) = {1/(nx)} - (1/n){1/x}, n>=2.

These functions vanish for x>1. On the cell 1/(j+1)<x<1/j,

    psi_n(x) = (j mod n)/n.                                   (6)

Indeed, writing 1/x=j+u, 0<u<1, gives floor((j+u)/n)=floor(j/n);
the terms u/n cancel. Endpoints are irrelevant to the L2 integral.

For a finite coefficient vector a=(a_2,...,a_N), define

    E_N(a)^2 = integral_0^1 |1-sum_{n=2}^N a_n psi_n(x)|^2 dx
             = sum_{j=1}^infinity
                 |1-sum_{n=2}^N a_n (j mod n)/n|^2 / [j(j+1)]. (7)

The cited theorem gives precisely

    RH  <=>  lim_{N->infinity} inf_a E_N(a) = 0.               (8)

To check the constraint does not strengthen the published criterion, take
F=sum_{n=1}^N b_n rho_n and A=sum b_n/n. On x>1, F=A/x, so
|A|<=||1_(0,1)-F||_2. Subtracting A rho_1 gives
F-A rho_1=sum_{n=2}^N b_n psi_n, with error at most
(1+||rho_1||_2)||1_(0,1)-F||_2. The reverse inclusion follows because each
psi_n is a linear combination of rho_n and rho_1. This proves (8) from
the cited criterion without assuming that any particular coefficients work.

The spectral bridge is the Mellin transform. For 0<Re(s)<1 the identity
integral_0^infinity rho_1(x)x^(s-1)dx=-zeta(s)/s is given in the same paper.
Scaling and subtracting give

    Mellin(psi_n)(s) = -zeta(s)(n^(-s)-n^(-1))/s.

Consequently Mellin-Plancherel at s=1/2+it turns (7) into

    E_N(a)^2 = (1/(2pi)) integral_{-infinity}^infinity
        |1+zeta(s)sum_{n=2}^N a_n(n^(-s)-n^(-1))|^2 / |s|^2 dt. (9)

Equations (7) and (9) are the same error in arithmetic-space and spectral
coordinates. This supplies a precise cross-domain connection; it does not
replace the convergence requirement with a visual resemblance.

For clarity, a hypothetical zeta zero rho=beta+i gamma with beta>1/2
would force E_N(a)^2 >= (2beta-1)/|rho|^2 for every N and a. At that zero
the Mellin transform of the error equals 1/rho. Cauchy-Schwarz bounds its
absolute value by E_N(a)/sqrt(2beta-1), proving the assertion. Thus driving
the full norm to zero really would exclude such zeros.

Unlike (H), criterion (8) is an equivalence to RH, without the stronger
all-coefficient/all-frequency requirement discussed in WEIGHTED_INTERACTIONS.md.

## 5. An exact finite-fit trap and a tail certificate [PROVEN]

For any finite a and J>=1, truncating (7) gives a rigorous interval:

    S_J(a) <= E_N(a)^2 <= S_J(a) + (1+sum_n|a_n|)^2/(J+1),     (10)

where S_J is its first J terms. This follows from 0<=psi_n<1 and
sum_{j>J}1/[j(j+1)]=1/(J+1). For rational a both endpoints are exact
rationals. This is an infinite-tail certificate for a given vector, not
a uniform convergence theorem for a changing family of vectors.

It is possible to fit every cell in any finite initial grid exactly.
For L>=2 put h_L=sum_{n<=L}mu(n)/n and choose

    a_n=-mu(n), 2<=n<=L;       a_(L+1)=(L+1)h_L.              (11)

For 1<=j<=L, substituting (6) cancels the terms j h_L, leaving
sum_{n<=j}mu(n)floor(j/n)=1. The last equality follows by summing
sum_{d|m}mu(d)=1_(m=1) over m<=j. Therefore S_L(a)=0 exactly for every L.

Yet at L=5 the coefficients are

    a_2=a_3=a_5=1, a_4=0, a_6=-1/5.

The first five cell errors vanish, while the sixth error equals 4/5.
That cell alone contributes

    E_6(a)^2 >= (4/5)^2/(6*7) = 8/525 > 0.                   (12)

This example does not determine the asymptotic behavior of (11). It proves
that exact fitting at an arbitrary finite grid cannot certify the full norm.
Proving (8) needs coefficient families for which the full tail vanishes too;
we have not established such a family. Simply sending J to infinity for
each fixed N computes its error, and does not prove that the optimized error
tends to zero as N grows.

## Structural checks [EMPIRICAL, WITH EXACT SUBCHECKS]

Run from the repository root with the existing environment:

```bash
python3 runtime/experiments/rft_mobius/check_arithmetic_routes.py
```

The script checks the divisor and prime-pairing identities through 512;
signed multiplicative controls and their convolution formula; exact rational
cell values and finite fits; rational tail intervals; and the exhaustive
q=8 independent-sign ensemble. Only the last check uses floating point.
The script does not extend the growth scan or modify saved probe results.
Finite checks catch algebra/implementation mistakes; the all-size conclusions
in this note rely on the displayed arguments and the cited criterion.

The audit has established identities, model bounds, and counterexamples to
insufficient inference rules. Neither deterministic Mobius concentration
nor convergence in (8) has been proved here.

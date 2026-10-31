# Analytic bounds for the Fibonacci-frame construction

This note derives bounds valid at every Fibonacci dimension. It does not
prove the Riemann hypothesis or boundedness of the experiment's normalized
records for every positive epsilon. It applies an existing stability argument
to the golden grid; no novelty claim is made.

## Definitions

Let F_0=0, F_1=1 and F_{r+1}=F_r+F_{r-1}. Put phi=(1+sqrt(5))/2 and
alpha=1/phi. For q=F_r and p=F_{r-1}, r>=2, use

    Phi_q[n,k] = q^(-1/2) exp(2 pi i n k alpha),
    n=0,...,q-1, k=1,...,q.

This is the experiment's raw matrix: frac(k phi) and frac(k alpha) agree.
Write G_q=Phi_q^* Phi_q and U_q=Phi_q G_q^(-1/2).

## 1. Uniform displacement of the golden grid [PROVEN]

The recurrence or Binet's formula gives the exact identity

    e_r := q alpha - p = (-1)^(r-1) phi^(-r).

The base case r=2 is alpha-1=-phi^(-2), and successive errors satisfy
e_{r+1}=-e_r/phi. Define delta=alpha-p/q=e_r/q. Since gcd(p,q)=1,
the residues j_k=kp mod q permute 0,...,q-1. Modulo integers,

    k alpha = j_k/q + k delta.

Subtract the common frequency c=(q+1)delta/2. In units of one grid spacing,
the remaining displacement is

    epsilon_{j_k} = (k-(q+1)/2)e_r,
    max_j |epsilon_j| = (q-1)phi^(-r)/2 < L_0 := 1/(2 sqrt(5)) < 1/4.

Here is the uniform inequality, including both parities. Set y=phi^(-r).
Binet gives q y=(1-(-1)^r y^2)/sqrt(5). For even r, (q-1)y is immediately
less than 1/sqrt(5). For odd r,

    (q-1)y = 1/sqrt(5) + y^2/sqrt(5) - y < 1/sqrt(5),

since 0<y<sqrt(5). This argument has no upper cutoff on r.

## 2. Uniform singular-value bounds [PROVEN, using published stability]

Yu and Townsend's discrete Kadec argument bounds the normalized perturbation
by rho(L)=1-cos(pi L)+sin(pi L). Their displayed matrices have odd size;
the following formulation explains why the argument also applies here for
even q.

Take t_n=2pi(n-(q-1)/2)/q and define

    Q[n,j]=q^(-1/2) exp(i j t_n),
    T[n,j]=q^(-1/2) exp(i (j+epsilon_j)t_n).

Q is unitary for every integer q. The scalar Fourier expansion of
1-exp(i epsilon t) into a constant, cos(ell t), and sin((ell-1/2)t)
has coefficient supremum sums, for |epsilon|<=L<1/4,

    1 - sin(pi L)/(pi L),
    sin(pi L)/(pi L) - cos(pi L),
    sin(pi L),

respectively. Multiplication by each sine/cosine diagonal has norm at most
one. Applying these expansions columnwise to Q and summing the operator
norm bounds gives ||T-Q||_2 <= rho(L). This uses orthogonality of Q and
|t_n|<=pi, with no odd-size restriction. This is the same scalar-series
argument as the cited theorem, with centered sampling for arbitrary q.

Column permutation, removing c, and centering n transform Phi_q into T
using only unitary diagonal matrices and a permutation. Thus their singular
values agree. Set

    d = cos(pi/(2 sqrt(5))) - sin(pi/(2 sqrt(5)))
      = 0.1171275880695115... > 0.

The perturbation inequality gives, for every Fibonacci dimension q,

    d <= sigma_min(Phi_q) <= sigma_max(Phi_q) <= 2-d,
    d^2 I <= G_q <= (2-d)^2 I,
    ||G_q^(-1/2)||_2 <= 1/d,
    kappa_2(G_q) <= ((2-d)/d)^2 = 258.4183706077... .

The Gram eigenvalue interval is approximately [0.01371887, 3.54520852].
This conservative bound proves uniform conditioning along Fibonacci sizes.
It does NOT prove the much tighter observed condition number near 2.8482,
and it does not cover arbitrary non-Fibonacci dimensions.

Source: A. Yu and A. Townsend, "On the stability of unevenly spaced samples
for interpolation and quadrature," Theorem 2.1 and its proof, followed by
the singular-value consequences. https://arxiv.org/abs/2202.04722
Published in BIT Numerical Mathematics (2023), DOI 10.1007/s10543-023-00965-z.

## 3. A raw prefix coefficient bound [PROVEN]

Let p_m contain m leading ones followed by zeros, with 1<=m<=q. Put
r_m=Phi_q^* p_m. The geometric-series formula gives

    |r_m[k]| <= q^(-1/2) min(m, 1/|sin(pi k alpha)|).

For q>=2, r>=3, the uncentered grid displacement obeys

    |k e_r| <= q phi^(-r) < 1/2.

For even r, Binet bounds the last expression by 1/sqrt(5); for odd r>=3
it is at most (1+phi^(-6))/sqrt(5)=0.47213595... .
For a residue j_k with circular distance h=min(j_k,q-j_k)>=1,

    ||k alpha||_(R/Z) >= (h-1/2)/q,
    |sin(pi k alpha)| >= 2(h-1/2)/q.

There is one residue at h=0 and at most two for each h>=1. Consequently,

    ||r_m||_1
      <= sqrt(q) [1 + sum_{h=1}^{floor(q/2)} 1/(h-1/2)]
      <= sqrt(q) (3+log q).

The final step uses the first summand 2 and bounds the remaining decreasing
sum by an integral. The case q=1 is immediate. The bound is uniform in both
the frame and the prefix cutoff, without a Mobius assumption.

## 4. Exact remaining arithmetic estimate [CONDITIONAL]

Define the signed raw-basis coefficient vector

    c_q = Phi_q^(-1) mu_q,
    mu_q = (mu(1),...,mu(q))^T.

For every prefix, exactly,

    M(m) = r_m^* c_q,
    |M(m)| <= sqrt(q)(3+log q) ||c_q||_infinity.       (A)

This is a different estimate from the plotted polar-coordinate absolute
overlap B_q(m); do not identify the two quantities.

A sufficient arithmetic hypothesis would be

    for every eta>0, there is D_eta independent of q such that
    ||Phi_q^(-1) mu_q||_infinity <= D_eta q^eta
    for every Fibonacci q.                          (H)

To check the implication carefully: choose q as the smallest Fibonacci
number at least m. Then q<2m (with q=m=1 harmless). For any epsilon>0,
apply (H) with eta=epsilon/2 and use

    log m <= m^(epsilon/2)/(e epsilon/2), m>=1.

Equation (A) yields |M(m)| <= C_epsilon m^(1/2+epsilon) for all m, with a
constant independent of m. The standard Mertens-growth characterization
would then imply RH. Hypothesis (H) is NOT proved here, and is asserted only
as sufficient, not as equivalent to RH. The further derivation in
[WEIGHTED_INTERACTIONS.md](WEIGHTED_INTERACTIONS.md) shows that (H) would
also imply uniform additive Mobius cancellation and GRH for Dirichlet
L-functions. It is a broader hypothesis than the untwisted bound needed
in the RH characterization; no converse implication is claimed.

What the proved conditioning bound actually supplies is only

    ||c_q||_infinity <= ||c_q||_2
      <= d^(-1)||mu_q||_2 <= d^(-1)sqrt(q).

It does not supply (H). A Mobius-specific cancellation argument is still
needed. Geometry cannot give (H) for arbitrary bounded input sequences:
for the constant input 1, taking m=q in (A) forces

    ||Phi_q^(-1) 1||_infinity >= sqrt(q)/(3+log q).

This does not refute (H) for Mobius; it isolates the need to use arithmetic.

## 5. What can be proved for the original plotted records [PROVEN]

For a cutoff m assigned to its smallest Fibonacci frame q, unitarity and
Cauchy-Schwarz give

    B_q(m) <= ||U_q^*p_m||_2 ||U_q^*mu_q||_2
           = sqrt(m sum_{n<=q} mu(n)^2)
           <= sqrt(mq) <= sqrt(2) m.

Therefore all the original normalized records are bounded for
epsilon>=1/2, by sqrt(2). For 0<epsilon<1/2 this argument gives only
sqrt(2)m^(1/2-epsilon), which grows. A growing upper estimate is not a proof
that the records themselves are unbounded; it is where this estimate stops.

Reference for the Mertens-growth/RH characterization: T. Tao, 254A Notes 2,
Exercise 33, https://terrytao.wordpress.com/2014/12/09/254a-notes-2-complex-analytic-multiplicative-number-theory/

## Numerical checks of the derivation [EMPIRICAL]

`python3 check_analytic_bounds.py` checks the Fibonacci identities in
100-digit decimal arithmetic through index 100, and compares the centered
and original matrices for even and odd sizes through 233. It also checks
the raw prefix l1 estimate and the raw-coefficient reconstruction identity.
These tests catch implementation and normalization mistakes; the all-size
claims above rely on the algebraic arguments, not the numerical checks.

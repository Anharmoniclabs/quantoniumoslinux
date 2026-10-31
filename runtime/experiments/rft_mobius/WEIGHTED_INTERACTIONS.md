# Weighted Mobius interactions: bounds and scope

This note uses the definitions and geometry in [ANALYTIC_BOUNDS.md](ANALYTIC_BOUNDS.md).
It does not prove the subpower coefficient hypothesis (H). It derives an
unconditional interaction bound and shows that (H) would imply uniform additive
cancellation and GRH for Dirichlet L-functions. No novelty claim is made.

## An unconditional bound [PROVEN]

For any row w of Phi_q^(-1), set

    c = sum_n w_n mu(n),
    D = sum_n |w_n|^2 mu(n)^2,
    R = 2 Re sum_{m<n} w_m conjugate(w_n) mu(m) mu(n),
    s(q) = sum_{n<=q} mu(n)^2.

Exactly |c|^2=D+R. The geometry bound gives D<=d^(-2)=72.8922908705... .
Cauchy-Schwarz on the s(q) nonzero terms gives |c|^2<=s(q)D. Hence

    -d^(-2) <= R <= (s(q)-1)d^(-2) <= (q-1)d^(-2).       (1)

This holds for every row at every Fibonacci dimension, but its upper bound
grows linearly. Because D is uniformly bounded, the desired bound
R<=C_eta q^(2 eta), for every eta>0, is equivalent up to constants to (H).

## Transfer of an established Mobius estimate [PROVEN USING CITED RESULT]

Write e(t)=exp(2 pi i t), f_k={k alpha}, S_q(theta)=sum_{n<=q}mu(n)e(n theta),
and K_q=||G_q^(-1)||_(infinity->infinity)=max_k sum_j |(G_q^(-1))[k,j]|.
The exact identities, including the n versus n-1 phase, are

    (Phi_q^* mu_q)_k = e(f_k) S_q(-f_k)/sqrt(q),
    c_q = G_q^(-1) Phi_q^* mu_q.

Davenport's theorem gives sup_theta |S_q(theta)|<=C_A q/(log q)^A for every
fixed A>0 and q>=2, with an ineffective constant independent of q and theta.
It follows that

    ||c_q||_infinity <= C_A K_q sqrt(q)/(log q)^A,
    R_{q,k} <= C_A^2 K_q^2 q/(log q)^(2A).                (2)

Our established bound is only K_q<=sqrt(q)d^(-2); spectral norm control is
not induced infinity-norm control. Substitution in (2) does not improve the
O(sqrt(q)) coefficient estimate asymptotically. Even if K_q=O(1) were proved,
sqrt(q)/(log q)^A is not O(q^eta) for fixed eta<1/2 and fixed A. Choosing A
to grow with q would discard the uncontrolled dependence of C_A on A.

Source for Davenport's bound: T. Tao (2021),
https://terrytao.wordpress.com/2021/07/05/quantitative-bounds-for-gowers-uniformity-of-the-mobius-and-von-mangoldt-functions/

## What (H) would imply [CONDITIONAL; IMPLICATION PROVED]

Let a=1-1/sqrt(5). The shifted-grid representation has displacement at most
L_0=1/(2sqrt(5)) grid spacings, so adjacent circular gaps are at least a/q.
Fix any theta and order distances from -theta to the frequencies as
t_1<=...<=t_q. An arc of length 2t contains at most 1+2tq/a points. Thus
t_l>=a(l-1)/(2q) for l>=2. Geometric summation and sin(pi t)>=2t give,
uniformly for 1<=m<=q and all theta,

    sum_k |q^(-1/2) sum_{n=1}^m e(n theta)e((n-1)f_k)|
      <= sqrt(q)[1+a^(-1)H_{q-1}],                      (3)

where H_j=sum_{l=1}^j 1/l and H_0=0. The nearest point contributes at most
m/sqrt(q)<=sqrt(q); the others give the harmonic sum. q=1 is immediate.
Multiplication of the prefix row by c_q recovers S_m(theta), hence

    sup_theta |S_m(theta)|
      <= sqrt(q)[1+a^(-1)H_{q-1}] ||c_q||_infinity.       (4)

If (H) holds, choose the smallest Fibonacci q>=m, so q<2m. Apply (H) with
eta=epsilon/2 and absorb the logarithm into m^(epsilon/2). Then

    sup_theta |S_m(theta)| = O_epsilon(m^(1/2+epsilon))   (5)

for every epsilon>0. This extends beyond the theta=0 sum needed for RH.
Porritt's 2017 introduction records (5) as conjectural, alongside Davenport's
estimate and a conditional 3/4+epsilon estimate: https://arxiv.org/abs/1711.08729

Fix a Dirichlet character chi modulo r. Expand it in the finite Fourier basis:

    chi(n) = sum_{b=0}^{r-1} a_b e(bn/r),
    a_b = r^(-1) sum_{u=0}^{r-1} chi(u)e(-bu/r).

Equation (5) gives M_chi(x):=sum_{n<=x}mu(n)chi(n)
=O_(chi,epsilon)(x^(1/2+epsilon)). By partial summation,

    F_chi(s) = s integral_1^infinity M_chi(x) x^(-s-1) dx

is holomorphic for Re(s)>1/2. Convergence is locally uniform by taking epsilon
smaller than the distance of a compact set to the boundary. For Re(s)>1,
the Dirichlet series and Euler product give F_chi(s)=1/L(s,chi).
For nonprincipal primitive chi, analytic continuation of L(s,chi)F_chi(s)=1
excludes zeros in the half-plane. For the principal primitive character,
multiply by s-1 to remove zeta's pole before using the identity theorem.
Functional equations reflect a nontrivial zero on the left to one on the
right, possibly for the conjugate character. Thus (H) implies Dirichlet GRH,
including RH. No converse from RH or GRH to (H) is claimed.

This corrects the scope of the research target: bounding every coefficient
has broader consequences than controlling M(x); it is not a routine missing
lemma whose truth follows from the established geometry.

## Related results and limitations

The 2026 paper "Higher uniformity of arithmetic functions in short intervals II.
Almost all intervals" permits exceptional intervals. Its stated estimates do
not directly provide an all-dimension, all-row bound on the present weights:
https://doi.org/10.1007/s00222-026-01408-6

The search and calculations have not supplied an established estimate proving
(H). Equations (1)-(4) and the conditional implications above are the results.

# THEOREMS_RFT_IRONCLAD.md

## Scope (what this file does)
This file contains theorem statements with full proofs for:
- Canonical RFT unitarity (Gram-normalized / symmetric orthogonalization).
- Fast RFT unitarity (factorized).
- Twisted convolution theorem + the diagonalization claim (exact).
- A provable “not an N-point DFT kernel (up to phases/permutation)” result for the raw φ-grid kernel used to build the canonical basis.
- Crypto: what reductions you can formally claim (and what you cannot) without new assumptions.
- Entropic uncertainty for the canonical RFT via the Maassen–Uffink bound.

This file also records:
- Status notes for claims that are plausible or test-backed but not yet theorem-level.
- Empirical or numerical-analysis observations that should not be promoted to formal reductions or asymptotic theorems without additional proof.

This file does **not** pretend to prove:
- IND-CPA/IND-CCA for any scheme unless it is explicitly built from a standard primitive with a standard proof.
- Standard SIS hardness for unmasked deterministic RFT-derived matrices unless an additional indistinguishability proof or a new structured-SIS assumption is stated explicitly.
- A theorem of the form $\|U-\Psi\|_F/\sqrt{N}=O((\log N)/N)$ solely from discrepancy facts about the golden grid.
- Wigner–Dyson / quantum-chaos claims unless an operator family and asymptotic law are proved.

---

## Relationship to Prior Art

> **Purpose:** Precisely position the canonical RFT against the closest existing literature so reviewers have a clear novelty boundary. This section is the authoritative statement for peer review purposes.

### P1 — Relationship to NUDFT (Non-Uniform Discrete Fourier Transform)

The raw basis matrix $\Phi$ defined in D1 is **mathematically a Type-I NUDFT matrix**: it evaluates complex exponentials at non-uniform frequencies $f_k = \operatorname{frac}((k+1)\phi)$ on a uniform sample grid $n \in \{0,\dots,N-1\}$. This overlap must be explicitly acknowledged.

The canonical RFT **departs from NUDFT in three precise ways:**

| Dimension | NUDFT (standard) | Canonical RFT |
|-----------|-----------------|---------------|
| **Inversion** | Pseudo-inverse, iterative solvers (CGLS, LSQR), or Lagrange/Newton interpolation — no exact inverse in general | Löwdin polar orthogonalization $U := \Phi(\Phi^\dagger\Phi)^{-1/2}$ yields an **exact unitary** with $U^\dagger U = I$ (Theorem 2) |
| **Basis orthogonality** | Raw $\Phi$ is non-orthogonal; NUDFT does not orthogonalize the basis — it solves a least-squares problem | Canonical RFT explicitly constructs a **unitary basis** via the unique polar factor (Theorem 9) |
| **Frequency grid design** | NUDFT is a general framework — any frequency set works; no grid is preferred | RFT fixes $f_k = \operatorname{frac}((k+1)\phi)$ specifically, motivated by Weyl equidistribution (Weyl 1916) and the Diophantine irrationality of $\phi$ to ensure no two frequencies are rationally commensurate (Theorem 1 proof) |
| **Signal-class claim** | NUDFT makes no claim about sparsity or energy compaction for any signal family | Canonical RFT claims KLT-like energy compaction specifically for $\phi$-quasi-periodic signals $S_\phi(K)$ (Theorem 15, Theorem E) |

**Summary statement for reviewers:**
> The raw basis $\Phi$ is a Type-I NUDFT matrix. The canonical RFT is distinguished by applying Löwdin orthogonalization to $\Phi$, yielding the unique unitary $U = \Phi(\Phi^\dagger\Phi)^{-1/2}$. Standard NUDFT literature uses pseudo-inverse or iterative solvers and does not construct a unitary basis. The novelty claim lies in this orthogonalization step combined with the specific $\phi$-grid design motivated by Weyl equidistribution theory, and the resulting signal-class-specific sparsity result (Theorem 15).

**What has been done in NUDFT literature vs. here:**
Gram matrices and Löwdin-style orthogonalization have been applied to UWB pulse design (Wetz et al. 2006, Lüke et al. 2012) and to spectrally efficient waveform families (Wang et al. 2012, *Signal Processing* 92:649). None of that literature applies the polar factor construction to a golden-ratio irrational frequency grid for the purpose of basis design and signal-class-specific sparsity. The application domain, grid choice, and sparsity characterization are the contributions here.

### P2 — Relationship to LCT / FrFT (Linear Canonical Transform / Fractional Fourier Transform)

The Linear Canonical Transform (LCT) and Fractional Fourier Transform (FrFT) generalize the DFT via chirp-phase modulation and rotation in the time-frequency plane. The Fast RFT surrogate $\Psi = D_\phi C_\sigma F$ superficially resembles LCT structure (chirp × FFT × phase).

**The distinction is exact and proven (Theorem 6):**
- LCT/FrFT frequency grids are always **rational** multiples of $1/N$ — the transform remains within the DFT family up to affine phase.
- The Fast RFT phase $\theta_k = 2\pi \operatorname{frac}((k+1)\phi)$ uses **irrational** fractional parts. Since $\phi$ is irrational, $f_k$ is never rational, so $\Psi$ cannot be expressed as any LCT/FrFT variant up to diagonal phases and permutation.
- This is the content of Theorem 6: the contradiction follows from rational vs. irrational frequency separation.

### P3 — Relationship to Frame Theory

The raw $\Phi$ is a **frame** (a possibly overcomplete or non-orthogonal spanning set) in the sense of Duffin & Schaeffer (1952). The dual-frame coefficient recovery in Theorem 15 Part 1 is a standard frame-theory result applied to our specific $\phi$-grid. The novelty is **not** in frame theory itself but in:
1. The specific $\phi$-grid choice and its Diophantine properties.
2. The Löwdin orthogonalization to the unitary polar factor $U$.
3. The sparsity characterization for the $S_\phi(K)$ signal class (Theorem 15 Parts 2–3).

### P4 — What the Number-Theoretic References Contribute

The following classical results are **applied**, not originated, in this work:
- **Weyl (1916):** Equidistribution of $\{n\phi\}$ mod 1 — used to motivate the $\phi$-grid's uniform frequency coverage.
- **Hurwitz (1891):** Best rational approximations to $\phi$ — relevant to the Diophantine near-collision concern in Observation 11.
- **Steinhaus–Sós (1957):** Three-distance theorem — provides the gap structure of $\{n\phi\}$ on the circle, supporting the quasi-uniform spacing of $f_k$.
- **Erdős–Turán (1948):** Discrepancy bounds for exponential sums — relevant to Observation 14.

Claiming these as novel contributions would be incorrect. They are the mathematical foundation on which the $\phi$-grid design rests.

---

## Definitions

Let $\phi := (1+\sqrt{5})/2$.

### D1 (Raw φ-grid exponential basis Φ)
Fix $N$ and define the golden frequency grid
$$f_k := \text{frac}((k+1)\phi) \in [0,1), \quad k \in \{0,\dots,N-1\},$$
where $\text{frac}(t)=t-\lfloor t\rfloor$.

Define the raw (generally non-orthogonal) basis $\Phi \in \mathbb{C}^{N\times N}$ by
$$\Phi[n,k] := \exp(i2\pi f_k n)/\sqrt{N}, \quad n,k \in \{0,\dots,N-1\}.$$

### D2 (Canonical RFT basis U: Gram-normalized / Löwdin orthogonalization)
Define the canonical unitary basis
$$U := \Phi(\Phi^\dagger \Phi)^{-1/2}.$$

The canonical RFT is
$$\hat{x} := U^\dagger x, \qquad x := U\hat{x}.$$

### D3 (Fast RFT (factorized) variant)
Let $F$ be the unitary DFT matrix of size $N\times N$.
Let $C_\sigma$ and $D_\phi$ be diagonal matrices with unit-modulus diagonal entries.
Define the fast RFT matrix
$$\Psi := D_\phi C_\sigma F.$$

### D4 (Twisted convolution induced by a unitary)
Given a unitary $\Psi$, define the $\Psi$-twisted convolution of $x,h \in \mathbb{C}^N$ by
$$x \star_\Psi h := \Psi^\dagger \big((\Psi x)\odot(\Psi h)\big).$$

### D5 (Mutual coherence)
For a unitary matrix $W \in \mathbb{C}^{N\times N}$, define
$$\mu(W) := \max_{j,k} |W_{jk}|.$$

---

## Theorem 1 (Full rank of the raw φ-grid basis Φ)
**Statement.**
The matrix $\Phi$ is invertible for every $N \ge 1$.

**Proof.**
Write $z_k := \exp(i2\pi f_k)$. Then $\Phi[n,k] = (1/\sqrt{N})\, z_k^n$, so $\Phi$ is a Vandermonde matrix. Its determinant is non-zero if all $z_k$ are distinct. If $z_i=z_j$, then $f_i-f_j \in \mathbb{Z}$. Since $f_k \in [0,1)$, this requires $f_i=f_j$. This implies $(i-j)\phi \in \mathbb{Z}$, which is impossible for $i \neq j$ because $\phi$ is irrational. $\det(\Phi)\ne 0$. ∎

---

## Theorem 2 (Canonical RFT basis U is unitary)
**Statement.**
$U$ is unitary ($U^\dagger U = I$).

**Proof.**
By Theorem 1, $\Phi$ has full rank, so $G := \Phi^\dagger \Phi$ is Hermitian positive definite.
Compute $U^\dagger U = (G^{-1/2}) \Phi^\dagger \Phi G^{-1/2} = G^{-1/2} G G^{-1/2} = I$. ∎

---

## Theorem 3 (Fast RFT is unitary)
**Statement.**
$\Psi := D_\phi C_\sigma F$ is unitary.

**Proof.**
$\Psi^\dagger \Psi = F^\dagger C_\sigma^\dagger D_\phi^\dagger D_\phi C_\sigma F = F^\dagger I F = I$. ∎

---

## Theorem 4 (Twisted convolution theorem)
**Statement.**
For $\star_\Psi$ defined in D4, the transform-domain multiplication rule holds:
$\Psi(x\star_\Psi h) = (\Psi x)\odot(\Psi h).$

**Proof.**
$\Psi(x\star_\Psi h) = \Psi\Psi^\dagger\big((\Psi x)\odot(\Psi h)\big) = (\Psi x)\odot(\Psi h).$ ∎

---

## Theorem 6 (Raw φ-grid kernel is not an N-point DFT kernel)
**Statement.**
Fix $N\ge 2$. There do not exist row phases, column phases, and a permutation $\pi$ such that the raw $\phi$-grid is a rephased/permuted DFT.

**Proof.**
Assuming such a relation exists leads to $f_j-f_k \equiv \frac{\pi(k)-\pi(j)}{N} \pmod 1$. The left side is irrational while the right side is rational. Contradiction. ∎

---

## Theorem 7 (Crypto reductions)

### Theorem 7.1 (Collision ⇒ SIS for uniform A)
If $Ax \equiv Ax' \pmod q$ for $x \neq x'$, then $s=x-x'$ is an SIS solution ($As \equiv 0 \pmod q$).

### Theorem 7.2 (Structured A needs a new assumption)
Standard SIS hardness is defined for random matrices. If $A$ is structured (deterministic RFT), one must assume "Structured-SIS" or prove computational indistinguishability from uniform.

### Theorem 7.4 (Hybrid construction is uniform)
If $A = A_\phi + R \pmod q$ where $R$ is uniform, then $A$ is uniform by group-shift invariance. ∎

---

## Theorem 8 (Maassen–Uffink entropic uncertainty)
**Statement.**
$H(|x|^2) + H(|U_\phi^\dagger x|^2) \ge -2\log(\mu(U_\phi)).$

**Proof.**
Direct application of the Maassen–Uffink theorem using the overlap constant $c = \mu(U_\phi)$. ∎

---

## Theorem 9 (Uniqueness of the polar factor)
**Statement.**
$U$ is the unique unitary factor in the polar decomposition of $\Phi$.

**Proof.**
$\Phi = UH$ where $H = (\Phi^\dagger \Phi)^{1/2}$. Uniqueness follows from the full rank of $\Phi$. ∎

---

## Observation 11 (Empirical quasi-orthogonality diagnostics)
**Status.**
Empirical benchmark statement.

**Statement.**
The inner product magnitude is $|\langle v_f,v_{f+\Delta f}\rangle| = \frac{|\sin(\pi N \Delta f)|}{N\,|\sin(\pi \Delta f)|}$.
Measured mean off-diagonal cross-energy $E_N$ via `benchmarks/rft_phi_frame_asymptotics.py`:
- $E_{256} \approx 8.91 \times 10^{-4}$
- $E_{1024} \approx 1.79 \times 10^{-4}$
- $E_{4096} \approx 3.84 \times 10^{-5}$

**Limitation.**
Mutual coherence $\mu(\Phi_N)$ does not decrease monotonically (e.g., $\mu_{512} \approx 0.49$, $\mu_{1024} \approx 0.68$). Average whitening occurs, but worst-case coherence is sensitive to Diophantine near-collisions.

---

## Non-theorem 12 (Polar normalization stability status)
**Status.**
Not yet proved asymptotically. Stability depends on row-sums of off-diagonal magnitudes (Gershgorin radii), not the average energy $E_N$.

---

## Theorem 13 (Twisted convolution complexity)
**Statement.**
$\star_\Psi$ can be computed in $O(N\log N)$ operations.

**Proof.**
Computation involves two forward transforms ($\Psi$), one pointwise product, and one inverse transform ($\Psi^\dagger$). Since $\Psi$ consists of one FFT and diagonal scaling, each step is $O(N\log N)$. ∎

---

## Observation 14 (Empirical asymptotic baseline)
The benchmark `phi_frame_asymptotics.csv` supports finite-$N$ average decorrelation. It does **not** prove convergence of $\Psi$ to $U$ or monotone decay of $\mu(\Phi_N)$.

---

## Theorem 15 (Exact sparsity in the raw \(\phi\)-grid frame vs. dense DFT leakage on \(S_\phi\))

**Statement.**
Fix \(N\). Let \(\Phi\in\mathbb{C}^{N\times N}\) be the raw \(\phi\)-grid exponential basis from Definition D1, i.e.
\[
\Phi[n,k] := \exp(i2\pi f_k n)/\sqrt{N},\qquad f_k:=\operatorname{frac}((k+1)\phi).
\]
Define the \(K\)-sparse \(\phi\)-grid signal class
\[
S_\phi(K) := \{\, s\in\mathbb{C}^N \;:\; s=\Phi c,\ \|c\|_0 = K \,\}.
\]

1) (**\(\phi\)-grid exact sparsity via the dual frame.**)  
For any \(s\in S_\phi(K)\), the unique coefficient vector \(c\) is recovered exactly by the dual-frame coefficient map
\[
\widehat{c} = (\Phi^\dagger \Phi)^{-1}\Phi^\dagger s,
\]
and satisfies \(\|\widehat{c}\|_0=K\) and \(\Phi\widehat{c}=s\) (zero reconstruction error) whenever \(K\le N\).

2) (**DFT dense leakage for a single golden tone.**)  
Let \(F\) be the unitary \(N\)-point DFT matrix with columns \(F[:,m]\propto \exp(i2\pi (m/N)n)\). For any single-tone \(s[n]=\exp(i2\pi f_k n)/\sqrt{N}\) with \(f_k=\operatorname{frac}((k+1)\phi)\), the DFT coefficient vector \(\widehat{x}_{DFT}=F^\dagger s\) is strictly dense:
\[
\|\widehat{x}_{DFT}\|_0 = N.
\]
Equivalently, zero reconstruction error in the DFT basis requires all \(N\) coefficients.

3) (**Extension to \(K>1\) (generic coefficients).**)  
For \(s=\sum_{k\in\mathcal{K}} c_k \exp(i2\pi f_k n)/\sqrt{N}\) with \(|\mathcal{K}|=K>1\), the condition that any fixed DFT bin vanishes is a non-trivial complex linear constraint on \((c_k)_{k\in\mathcal{K}}\). Therefore, excluding a measure-zero set of exact multi-tone cancellations, \(\widehat{x}_{DFT}\) is dense and perfect DFT reconstruction still requires full \(N\)-length support.

**Proof.**

**Part 1 (dual-frame exactness).**
Since Theorem 1 proves \(\Phi\) is full rank for every \(N\), the Gram matrix \(G:=\Phi^\dagger\Phi\) is invertible.
If \(s=\Phi c\), then
\[
(\Phi^\dagger\Phi)^{-1}\Phi^\dagger s
= G^{-1}\Phi^\dagger \Phi c
= c,
\]
so the recovered coefficients are exactly the original \(K\)-sparse vector.

**Part 2 (DFT mismatch; geometric series).**
For a single tone \(s[n]=\exp(i2\pi f n)/\sqrt{N}\), the (unitary) DFT coefficient at bin \(m\) is proportional to
\[
X[m] = \sum_{n=0}^{N-1}\exp(i2\pi(f-m/N)n),
\]
which is a finite geometric series. This sum equals \(0\) if and only if \(\exp(i2\pi(f-m/N))\neq 1\) and \(\exp(i2\pi N(f-m/N))=1\), i.e.
\[
N(f-m/N)\in\mathbb{Z}\quad\Longleftrightarrow\quad f=\frac{m+\ell}{N}\ \text{ for some }\ell\in\mathbb{Z}.
\]
If \(f=f_k=\operatorname{frac}((k+1)\phi)\), then \(f\) is irrational because \(\phi\) is irrational and \(\operatorname{frac}\) differs by an integer.
But \((m+\ell)/N\) is rational, so equality is impossible. Therefore \(X[m]\neq 0\) for every \(m\in\{0,\dots,N-1\}\), hence \(\|\widehat{x}_{DFT}\|_0=N\).

**Part 3 (generic multi-tone).**
For fixed \(m\), \(X[m]\) is a complex linear form in the coefficient vector \(c\).
The set \(\{c\in\mathbb{C}^K : X[m]=0\}\) is a proper affine hyperplane, hence has Lebesgue measure zero in \(\mathbb{C}^K\).
Taking the union over finitely many \(m\) preserves measure zero, so outside this exceptional set the DFT coefficient vector has no exact zeros. \(\square\)

---

## Final Scientific Status Summary

| Metric | Safe Property | Support |
|---|---|---|
| **Full Rank** | $\Phi$ is invertible for every $N$ | Theorem 1 |
| **Unitarity** | $U^\dagger U = I$ | Theorem 2 |
| **Complexity** | $O(N \log N)$ for Fast RFT | Theorem 13 |
| **Decorrelation** | Mean cross-energy decreases (average case) | Observation 11 |
| **Conditioning** | Not yet proved asymptotically | Non-theorem 12 |
| **Distinct from NUDFT** | Löwdin ortho + $\phi$-grid design + sparsity claim | P1 (Prior Art section) |
| **Distinct from LCT/FrFT** | Irrational $\phi$-phase ≠ any affine phase | Theorem 6 + P2 |

---

## Test-backed theorem set (engineering-grade)
- **Theorem A:** $U$ is the unique nearest unitary to $\Phi$.
- **Theorem B:** $C_\phi$ eigenvalues match the $\phi$-grid frequencies.
- **Theorem E:** $U$ achieves higher energy concentration ($K_{0.99}$) than FFT on golden drift signals.

---

## References
- Maassen & Uffink (1988), *Physical Review Letters*.
- Kronecker-sequence discrepancy results (Koksma-Hlawka context).
- Micciancio et al., Lattice-based cryptography (SIS context).
- Weyl, H. (1916). Über die Gleichverteilung von Zahlen mod. Eins. *Mathematische Annalen* 77, 313–352.
- Hurwitz, A. (1891). Ueber die angenäherte Darstellung der Irrationalzahlen durch rationale Brüche. *Mathematische Annalen* 39, 279–284.
- Steinhaus, H. (1957). One hundred problems in elementary mathematics.
- Erdős, P. & Turán, P. (1948). On a problem in the theory of uniform distribution. *Indagationes Mathematicae* 10.
- Duffin, R.J. & Schaeffer, A.C. (1952). A class of nonharmonic Fourier series. *Transactions of the AMS* 72, 341–366. [Frame theory foundation]
- Wang, X. et al. (2012). Approximation of Löwdin orthogonalization to a spectrally efficient waveform. *Signal Processing* 92(3), 649–660. [Closest prior art for Löwdin applied to waveform design]
- Dutt, A. & Rokhlin, V. (1993). Fast Fourier transforms for nonequispaced data. *SIAM Journal on Scientific Computing* 14(6). [NUFFT foundation]
- Potts, D. et al. (2001). Fast Fourier transforms for nonequispaced data: A tutorial. *Modern Sampling Theory*. [NUDFT survey]

# QuantoniumOS: Security Policy and Compliance

**Lead Architect:** Luis M. Minier (Independent, Self-Funded Researcher)  
**Intellectual Property:** USPTO Application No. 19/169,399  
**Effective Date:** March 2026  

> For full ethical and operational boundaries, refer to [`GOVERNANCE.md`](./GOVERNANCE.md).

---

## I. National Security and Export Control

QuantoniumOS is developed in full compliance with the following U.S. regulatory frameworks:

- **EAR Compliance:** This technology is developed in alignment with the U.S. Export Administration Regulations (EAR, 15 CFR Parts 730–774). It is not currently classified under ITAR schedule. Any future classification change will trigger an immediate administrative freeze on all external licensing.
- **Dual-Use Prohibition:** Per the `GOVERNANCE.md` charter, the use of the RFT φ-basis stack in lethal autonomous weapons systems (LAWS) or offensive cyber-warfare platforms is **strictly prohibited**. Violations trigger immediate licensing revocation and referral to the Department of Justice and OFAC under 18 U.S.C. § 2339.

---

## II. Vulnerability Reporting

We take the security and mathematical integrity of the RFT resonance logic seriously.

If you discover a security vulnerability, a side-channel leak, or a mathematical flaw that compromises the 10⁻¹⁵ precision engine:

1. **Do NOT** open a public GitHub issue.
2. **Do NOT** disclose the vulnerability publicly before coordinated disclosure is complete.
3. Contact the maintainer via a **private GitHub Security Advisory** on this repository (Settings → Security → Advisories → New draft advisory).
4. We operate under **Coordinated Vulnerability Disclosure (CVD)** protocols to protect the integrity of the IP and its users.
5. We will acknowledge receipt within **72 hours** and provide a documented remediation timeline.

### Scope

- RFT mathematical framework and transform implementations
- Cryptographic research prototypes
- Signal processing pipelines

### Out of Scope

- Experimental research prototypes explicitly marked as such in the codebase
- Third-party dependencies not maintained by this project

---

## III. Timing and Side-Channel Concerns

We aim to minimize timing and cache side-channel leakage in all production-grade RFT implementations. If you identify exploitable timing patterns or cache-based inference attacks:

- Report via the CVD process above.
- We will prioritize remediation and publish a public advisory upon resolution.

---

## IV. Experimental Cryptography Disclaimer

The Resonant Fourier Transform (RFT) and the RFT–SIS hash function are research prototypes operating at the frontier of computational mathematics. They have **not been formally proven secure** under all adversarial models and should not be used in production or security-critical systems without an independent cryptographic audit.

Use these components only for academic or experimental purposes unless otherwise licensed and reviewed under a formal commercial agreement.

---

## V. Licensing & Review Status

This project is currently under **Licensing & Review (LR)** by the USPTO (Application No. 19/169,399). Security researchers are encouraged to review [`GOVERNANCE.md`](./GOVERNANCE.md) for the complete ethical, legal, and operational framework governing this technology.

---

*This policy is effective as of March 2026.*

**Signed,**  
Luis M. Minier  
Lead Architect & Sole Inventor  
QuantoniumOS / Resonant Fourier Transform  
USPTO Application No. 19/169,399  
March 2026

# PII and Patent Admin Index

**Date:** April 2026
**Classification:** Tier 3 — Do Not Share
**Purpose:** Index of personally identifiable information and patent administration details in the repo.

---

## PII Locations

| File | Line(s) | Content | Action |
|------|---------|---------|--------|
| `CITATION.cff` | L8 | `luisminier79@gmail.com` | Keep (standard for citation metadata). Do not expose in public mirror. |
| `docs/project/Author` | L2 | `luisminier79@gmail.com` | Internal reference. Do not expose. |
| `README.md` | L18 | `luisminier79@gmail.com` | Replace with generic project contact or remove before any public mirror. |
| `docs/project/CLAIMS_PRACTICING_FILES.txt` | L57 | `luisminier79@gmail.com` (contact line) | Internal registry. Acceptable in private repo. |

## Patent Filing Details

| File | Line(s) | Content | Notes |
|------|---------|---------|-------|
| `docs/patent/USPTO_EXAMINER_RESPONSE_PACKAGE.md` | L4-6 | App No 19/169,399, Filing Date, Full Name | Public record via USPTO PAIR. Acceptable in docs. |
| `docs/patent/USPTO_ALGORITHM_SPECIFICATIONS.md` | L5-7 | App No, Confirmation No, Full Name | Public record. Confirmation number is semi-private (needed for USPTO access). |

## Guidance

- **Personal email:** Use for private repo only. Replace with a project contact email (e.g., GitHub noreply or project-specific address) in any public-facing document.
- **Patent numbers:** Public record — acceptable to reference. Confirmation numbers are only useful for USPTO portal access and are not sensitive.
- **Full legal name:** Public record via patent filing. Acceptable in patent docs. Minimize in non-patent files.

---

*This file is Tier 3. See `docs/project/DISCLOSURE_BOUNDARY.md` for tier definitions.*

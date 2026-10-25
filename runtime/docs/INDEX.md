# RFT Research Framework — Documentation Index

**Updated:** March 2026
**Purpose:** Navigate the project documentation, starting from the core thesis.

---

## Start Here

| Doc | Purpose |
|-----|---------|
| **[../README.md](../README.md)** | Project overview — includes "Frequency as Infrastructure" unified thesis (STORE / COMPUTE / COMPOSE / ROUTE) |
| **[../ENGINEER_MANUAL.md](../ENGINEER_MANUAL.md)** | Full engineering manual v2.1.0 — Executive Layer, Architecture, Theorems, Performance |
| **[technical/ARCHITECTURE_OVERVIEW.md](technical/ARCHITECTURE_OVERVIEW.md)** | Architecture overview with unified design philosophy |

## Core Science

| Doc | Purpose |
|-----|---------|
| **[THEOREMS_RFT_IRONCLAD.md](THEOREMS_RFT_IRONCLAD.md)** | Formal proofs — Theorems 1–15 + Theorem E (unitarity, non-equivalence to DFT, golden spectral concentration, RIP) |
| **[MATHEMATICAL_CLAIMS_INVENTORY.md](MATHEMATICAL_CLAIMS_INVENTORY.md)** | Complete claims inventory with proof status for every theorem, lemma, and conjecture |
| **[NON_CLAIMS.md](NON_CLAIMS.md)** | What RFT does NOT claim — required reading |
| **[LIMITATIONS_AND_REVIEWER_CONCERNS.md](LIMITATIONS_AND_REVIEWER_CONCERNS.md)** | Pre-answered reviewer questions |
| **[GLOSSARY.md](GLOSSARY.md)** | General term definitions |
| **[RFT_TERMINOLOGY_GLOSSARY.md](RFT_TERMINOLOGY_GLOSSARY.md)** | **RFT-specific terminology** — Canonical vs Fast vs Fibonacci (MUST READ for agents) |

## RFT Variants & Algorithms

| Doc | Purpose |
|-----|---------|
| **[NOVEL_ALGORITHMS.md](NOVEL_ALGORITHMS.md)** | Full algorithm inventory (36 distinct implementations) |
| **[../algorithms/rft/README_RFT.md](../algorithms/rft/README_RFT.md)** | Canonical RFT definition + 14 variants |
| **[HYBRID_FFT_RFT_ALGORITHM.md](HYBRID_FFT_RFT_ALGORITHM.md)** | O(N log N) hybrid factorization |
| **[../docs/project/CANONICAL.md](../docs/project/CANONICAL.md)** | Which code is claim-bearing vs exploratory |

## Validation & Benchmarks

| Doc | Purpose |
|-----|---------|
| **[../BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md)** | Benchmark methodology |
| **[benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md](benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md)** | GPT-2 KV-cache head-dim compression: RFT Auto **beats baseline by 10.2%**, beats DCT by 5.67 PPL (March 2026, verified) |
| **[papers/kv_cache_intelligence_dividend_2026.md](papers/kv_cache_intelligence_dividend_2026.md)** | Zenodo paper: The Intelligence Dividend — non-circular KV-cache benchmark across 3 models, 2 architectures |
| **[research/RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md](research/RESEARCH_SOURCES_AND_ANALYSIS_GUIDE.md)** | External validation sources |

---

## Supporting Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [LICENSE.md](LICENSE.md) | AGPL-3.0-or-later | Everyone |
| [LICENSE-CLAIMS-NC.md](LICENSE-CLAIMS-NC.md) | Non-commercial claims | Everyone |
| [PATENT_NOTICE.md](PATENT_NOTICE.md) | Patent information | Commercial users |
| [CLAIMS_PRACTICING_FILES.txt](CLAIMS_PRACTICING_FILES.txt) | Patent file list | Developers |
| [PATENT_COMPLIANCE_REPORT.md](PATENT_COMPLIANCE_REPORT.md) | Compliance status | Legal review |
| [SECURITY.md](SECURITY.md) | Security policy | Security researchers |

---

## Quick Navigation by Task

### I want to...

#### Understand the System
1. Start with [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) - Quick overview
2. Read [README.md](README.md) - Project introduction
3. Dive into [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) - Complete details

#### Get Started
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) - First steps
2. Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation
3. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands

#### Clean Up the Repository
1. Review [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) - Understand structure
2. Follow [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md) - Step-by-step guide
3. Use [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md) - Copy-paste commands

#### Run Benchmarks
1. Check [COMPETITIVE_BENCHMARK_RESULTS.md](COMPETITIVE_BENCHMARK_RESULTS.md) - Previous results
2. Run `python benchmarks/kv_cache_perplexity_benchmark.py --retain 0.40` — KV-cache head-dim perplexity ([results](benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md))
3. Run `python benchmarks/kv_cache_boundary_error.py` — Positional encoding leakage
4. Follow [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md) - Reproducibility

#### Develop Features
1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture
2. Check [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) - Find relevant code
3. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Dev commands

#### Fix Bugs
1. Check [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) - Known issues
2. Review [TEST_RESULTS.md](TEST_RESULTS.md) - Test status
3. Use [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md) - Testing commands

#### Contribute
1. Read [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) - Organization
2. Check [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md) - Priority tasks
3. Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) - Development setup

---

## 📁 Documentation by Category

### System Overview
- ✨ [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) - **NEW: Complete system map**
- ✨ [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) - **NEW: Quick status**
- [README.md](README.md) - Project overview
- [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) - Organization summary

### Setup & Installation
- [GETTING_STARTED.md](GETTING_STARTED.md) - First steps
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation guide
- [quantoniumos-bootstrap.sh](quantoniumos-bootstrap.sh) - Automated setup
- [verify_setup.sh](verify_setup.sh) - Verification script

### Development
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Developer commands
- ✨ [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md) - **NEW: Copy-paste commands**
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture
- [docs/ARCHITECTURE_QUICKREF.md](docs/ARCHITECTURE_QUICKREF.md) - Quick reference

### Maintenance
- ✨ [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md) - **NEW: Cleanup guide**
- [organize-release.sh](organize-release.sh) - Release packager
- [scripts/validate_all.sh](scripts/validate_all.sh) - Full validation

### Testing & Validation
- [TEST_RESULTS.md](TEST_RESULTS.md) - Test execution status
- [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md) - Reproducibility guide
- [ARCHITECTURE_VERIFICATION.md](ARCHITECTURE_VERIFICATION.md) - Architecture validation
- [hardware/HW_TEST_RESULTS.md](hardware/HW_TEST_RESULTS.md) - Hardware validation

### Performance & Benchmarks
- [COMPETITIVE_BENCHMARK_RESULTS.md](COMPETITIVE_BENCHMARK_RESULTS.md) - Industry comparison
- [benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md](benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md) - **KV-cache perplexity: RFT Auto PPL=29.85, beats DCT (35.51) and baseline (33.26)**
- [benchmarks/run_all_benchmarks.py](benchmarks/run_all_benchmarks.py) - Benchmark runner
- [ROUTING_OPTIMIZATION.md](ROUTING_OPTIMIZATION.md) - Routing performance

### Research & Theory
- [docs/validation/RFT_THEOREMS.md](docs/validation/RFT_THEOREMS.md) - Mathematical proofs
- [papers/](papers/) - Academic papers
- [experiments/](experiments/) - Research experiments
- [docs/research/](docs/research/) - Research documentation

### Legal & Licensing
- [LICENSE.md](LICENSE.md) - AGPL-3.0-or-later
- [LICENSE-CLAIMS-NC.md](LICENSE-CLAIMS-NC.md) - Non-commercial license
- [PATENT_NOTICE.md](PATENT_NOTICE.md) - Patent information
- [PATENT_COMPLIANCE_REPORT.md](PATENT_COMPLIANCE_REPORT.md) - Compliance report
- [SECURITY.md](SECURITY.md) - Security policy

---

## Finding Specific Information

### Code Implementation
**Question:** Where is the optimized RFT implementation?  
**Answer:** [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) → "Core Algorithm Components" → See `algorithms/rft/core/rft_optimized.py`

### Benchmark Results
**Question:** How does Φ-RFT compare to FFT/DCT?
**Answer:** [COMPETITIVE_BENCHMARK_RESULTS.md](COMPETITIVE_BENCHMARK_RESULTS.md) → "Class B: Transform & DSP"
Or [benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md](benchmarks/KV_CACHE_PERPLEXITY_BENCHMARK.md) → GPT-2 KV-cache perplexity (RFT Auto wins, -10.2% vs baseline)
Or [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) → "Performance Summary"

### Known Issues
**Question:** What's broken?  
**Answer:** [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) → "Known Issues"

### Cleanup Tasks
**Question:** What can I safely delete?  
**Answer:** [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md) → "Phase 1: Safe Deletions"

### Test Status
**Question:** Which tests are passing?  
**Answer:** [TEST_RESULTS.md](TEST_RESULTS.md) or [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) → "Tests"

### Hardware Status
**Question:** Is hardware simulation working?  
**Answer:** [hardware/HW_TEST_RESULTS.md](hardware/HW_TEST_RESULTS.md) or [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) → "Hardware Implementations"

---

## Documentation Statistics

### Coverage Analysis
- **Total Documentation Files:** 307 (md + txt)
- **Root-Level Guides:** 25+
- **Technical Deep Dives:** 10+
- **API Documentation:** Pending (Sphinx)
- **Video Tutorials:** Planned

### New Documentation (This Scan)
1. [OK] **SYSTEM_ARCHITECTURE_MAP.md** - 850+ lines, complete system map
2. [OK] **CLEANUP_ACTION_PLAN.md** - 600+ lines, actionable cleanup
3. ✅ **SYSTEM_STATUS_SUMMARY.md** - 400+ lines, quick reference
4. ✅ **CLEANUP_COMMANDS.md** - 500+ lines, command reference
5. ✅ **DOCUMENTATION_INDEX.md** - This file

**Total New Documentation:** 2,750+ lines

---

## 🎓 Learning Paths

### For New Users
1. [README.md](README.md) - Understand the project
2. [GETTING_STARTED.md](GETTING_STARTED.md) - First steps
3. [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) - Current status
4. [docs/ARCHITECTURE_QUICKREF.md](docs/ARCHITECTURE_QUICKREF.md) - Quick reference

### For Developers
1. [SETUP_GUIDE.md](SETUP_GUIDE.md) - Development setup
2. [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md) - System deep dive
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture
4. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
5. [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md) - Development tasks

### For Researchers
1. [README.md](README.md) - Project overview
2. [COMPETITIVE_BENCHMARK_RESULTS.md](COMPETITIVE_BENCHMARK_RESULTS.md) - Results
3. [docs/validation/RFT_THEOREMS.md](docs/validation/RFT_THEOREMS.md) - Proofs
4. [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md) - Reproduce experiments
5. [papers/](papers/) - Academic papers

### For Maintainers
1. [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md) - Current status
2. [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md) - Maintenance tasks
3. [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md) - Maintenance commands
4. [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) - Organization
5. [TEST_RESULTS.md](TEST_RESULTS.md) - Test status

---

## 🔗 External Resources

### Academic Publications
- **Zenodo:** https://doi.org/10.5281/zenodo.17712905
- **TechRxiv:** https://doi.org/10.36227/techrxiv.175384307.75693850/v1
- **Coherence Paper:** https://zenodo.org/records/17726611
- **RIP-First Architecture (March 2026):** [docs/papers/rip_first_architecture_2026.md](papers/rip_first_architecture_2026.md) -- Operator-only IHT sparse recovery, 27-variant benchmark

### GitHub Resources
- **Repository:** https://github.com/mandcony/quantoniumos
- **Issues:** https://github.com/mandcony/quantoniumos/issues
- **Discussions:** https://github.com/mandcony/quantoniumos/discussions

### Patent Information
- **USPTO Application:** 19/169,399
- **Status:** Patent Pending
- **Filed:** April 3, 2025

---

## 📞 Support & Contact

### Documentation Issues
If you find any documentation issues:
1. Check this index for correct document
2. Search the document for your topic
3. File an issue on GitHub
4. Contact author directly

### Author Contact
**Luis M. Minier**  
📧 luisminier79@gmail.com  
🐙 https://github.com/mandcony

### Community
- GitHub Issues - Bug reports
- GitHub Discussions - Questions & ideas
- Direct email - Commercial licensing

---

## Next Steps

After reading this index:

1. **New Users:**  
   → [GETTING_STARTED.md](GETTING_STARTED.md)

2. **Understanding System:**  
   → [SYSTEM_ARCHITECTURE_MAP.md](SYSTEM_ARCHITECTURE_MAP.md)

3. **Cleaning Repository:**  
   → [CLEANUP_ACTION_PLAN.md](CLEANUP_ACTION_PLAN.md)

4. **Running Commands:**  
   → [CLEANUP_COMMANDS.md](CLEANUP_COMMANDS.md)

5. **Quick Status Check:**  
   → [SYSTEM_STATUS_SUMMARY.md](SYSTEM_STATUS_SUMMARY.md)

---

## Document Maintenance

### Keeping This Index Updated

When adding new documentation:
1. Add entry to appropriate category
2. Update "Documentation Statistics"
3. Add to relevant "Learning Path"
4. Update navigation section if needed

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 3, 2025 | Initial index after complete system scan |

---

**End of Documentation Index**

*This index organizes all quantoniumos documentation for easy navigation.*  
*Update this file whenever major documentation changes occur.*

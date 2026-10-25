# Stub and Legacy Files — Honest Inventory

> **Updated March 2026** — The four root-level `quantum_*.py` stub files
> (`quantum_conversation_manager.py`, `quantum_rlhf_system.py`,
> `quantum_domain_fine_tuner.py`, `quantum_safety_system.py`) have been
> **deleted**. They contained only license headers with zero code — no
> classes, no functions, no imports. The chatbox GUI
> (`src/apps/qshll_chatbox.py`) already guards its imports with
> `try/except ImportError` and sets `QUANTUM_AI_AVAILABLE = False` when
> the files are absent, so removing them has no runtime impact.

---

## Historical context

These files originally carried legacy `quantum_*` names and were described
as compatibility shims for the chatbox GUI. In practice they were emptied
to license-header-only stubs during a prior cleanup and no longer provided
any importable symbols. They have been removed to reduce confusion.

---

*Last updated: March 2026*

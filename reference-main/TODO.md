# TODO.md — QUE Guild Godot Frontend + Python Backend + Research Conclave

## Objective

Transform QUE Guild from a rough Python-rendered prototype into a **safe, efficient, local command-center app** with:

- **Python backend** as the source of truth
- **Godot frontend** as the visual game layer
- **Research Conclave mode** for multi-agent task decomposition and synthesis
- **Read-only-by-default repo operations**
- **Mini Bronx neighborhood command center aesthetic**

This build must remain legacy-safe, deterministic, auditable, and useful for real project work.

---

## Core Product Rules

- Python owns all repo inspection, routing, quest execution, reporting, and safety controls.
- Godot owns all visuals, UI, animation, neighborhood map rendering, and user interaction.
- Godot must **not** directly inspect the repo.
- Agents must **not** autonomously edit source code.
- Allowed writes are limited to:
  - `data/agent_state/`
  - `data/agent_reports/`
  - backend cache/export state
  - logs
- Skill execution must run through the skill library.
- Structured results are mandatory.
- Research Conclave must be evidence-based, not fake agent chatter.

---

## Phase 0 — Audit and Freeze Current State

### Goal
Understand current implementation and freeze a known-good migration baseline.

### Tasks
- [x] Create a migration branch for the QUE Guild rebuild.
- [x] Inventory current backend files
- [x] Document which files are reusable / deprecated / scaffolding
- [x] Record current known issues
- [x] Write migration notes file: `docs/que_guild_migration_notes.md`

### Acceptance
- [x] Current system status documented
- [x] Migration branch exists
- [x] Known defects captured before refactor begins

---

## Phase 1 — Backend Data Model Hardening

### Goal
Make the backend structurally real before touching presentation.

### 1.1 Add structured result models

- [ ] Create `QuestResult` dataclass
- [ ] Create `ResearchPacket` dataclass
- [ ] Create `WorldState` schema model
- [ ] Create `QuestRecord` schema model
- [ ] Create `AgentStateView` schema model
- [ ] Ensure all models are JSON serializable

### 1.2 Eliminate raw string-only quest outputs

- [ ] Update all agent skills to return `QuestResult`
- [ ] Preserve human-readable summary strings inside the structured object
- [ ] Update quest board persistence to store structured results

### Acceptance
- [ ] No main skill returns plain strings as its primary result
- [ ] Quest results can be serialized cleanly to JSON
- [ ] Agent outputs now support severity/confidence/evidence

---

## Phase 2 — Make SkillLibrary the Real Execution Backbone

- [ ] Replace direct `_skill_*` invocation with `skill_library.execute()`
- [ ] Add real execution metrics (times_used, avg_duration_ms, success_rate)
- [ ] Ensure failures are recorded as structured failed QuestResults
- [ ] Add unit tests for skill execution telemetry

---

## Phase 3 — ORACLE Routing and Patrol Modes

- [ ] Add changed-path routing classifier
- [ ] Implement patrol modes: scout, patrol, siege, release_gate
- [ ] Add contradiction detection layer
- [ ] Persist patrol mode and timestamps across sessions

---

## Phase 4 — Research Conclave Mode

- [ ] Add research_conclave top-level mode
- [ ] ORACLE decomposes parent task into agent-scoped subtasks
- [ ] Agents return ResearchPackets
- [ ] Optional second-pass follow-up with depth limit
- [ ] ORACLE synthesizes final report with attribution

---

## Phase 5 — Backend API Bridge (FastAPI)

- [ ] Add local FastAPI app with /health, /state/*, /action/* endpoints
- [ ] Define stable response schemas
- [ ] Ensure backend runs headless

---

## Phase 6 — Backend Safety Fixes

- [ ] Fix WARD dependency audit classification
- [ ] Reduce secret-scan false positives
- [ ] Expand LORE stale scan beyond root markdown
- [ ] Fix HUD state field mismatch

---

## Phases 7–15 — Godot Frontend (deferred)

See full phase descriptions above. Backend must be solid first.

---

## Immediate Priority Order

1. [ ] Add structured models (QuestResult, ResearchPacket, WorldState)
2. [ ] Route skill execution through SkillLibrary.execute()
3. [ ] Refactor quest persistence to structured results
4. [ ] Implement ORACLE routing and patrol modes
5. [ ] Implement Research Conclave
6. [ ] Expose backend state through FastAPI
7. [ ] Create Godot project skeleton
8. [ ] Connect Godot to backend state
9. [ ] Build Bronx block world
10. [ ] Build command HUD
11. [ ] Integrate patrol/research actions
12. [ ] Add atmosphere and polish
13. [ ] Add launch scripts
14. [ ] Finish docs

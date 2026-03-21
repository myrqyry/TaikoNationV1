# TaikoNation Consolidation Roadmap

This roadmap converts the current "parallel modernization" state into a single maintainable architecture.

## Target outcomes

- One backend API surface (FastAPI).
- One persistent data model (SQLite first, Postgres-ready).
- One job execution path for long-running ML tasks.
- One canonical package/CLI/documentation workflow.

## Phase 0 — Baseline and guardrails (1–2 days)

### Goals

- Freeze current behavior before migration.
- Add enough checks to safely remove duplicate systems.

### Checklist

- [ ] Record endpoint inventory from `web/server.py` and `web/server_fastapi.py`.
- [ ] Tag each endpoint as: **keep**, **merge**, **drop**, or **deprecate**.
- [ ] Add migration-tracking table in repo docs (`old endpoint` → `new endpoint` → `status`).
- [ ] Add smoke tests for high-risk flows: upload, generate, export, evaluate, websocket progress.

## Phase 1 — Single backend (FastAPI-only) (3–5 days)

### Goals

- Stop dual-backend drift.
- Preserve useful Flask features via explicit migration.

### Checklist

- [ ] Keep `web/server_fastapi.py` as the primary app entrypoint.
- [ ] Port only high-value Flask features from `web/server.py`:
  - Task lifecycle endpoints.
  - Validation and editor-related APIs that are actually used by the UI.
  - Config read/write endpoints with strict schema validation.
- [ ] Add explicit `/api/v1` route grouping and response models.
- [ ] Add deprecation notice in Flask server module header.
- [ ] Remove Flask from runtime path after parity check passes.

### Exit criteria

- Frontend works entirely against FastAPI routes.
- No production/runtime path imports `web/server.py`.

## Phase 2 — Persistence layer (3–4 days)

### Goals

- Replace in-memory demo state with durable records.

### Checklist

- [ ] Add SQLAlchemy 2.x or SQLModel data layer under `taikonation/`.
- [ ] Start with SQLite file DB (local dev), keep Postgres-ready DSN config.
- [ ] Create minimal tables:
  - `models`
  - `tasks`
  - `generations`
  - `evaluations`
  - `artifacts`
- [ ] Move mutable state out of module globals and into repositories/services.
- [ ] Add migration tool (`alembic`) and first migration.

### Exit criteria

- Restart does not erase jobs/results.
- API state reads from DB, not process memory.

## Phase 3 — Unified async jobs (3–4 days)

### Goals

- Make long-running ML work robust, cancellable, and observable.

### Checklist

- [ ] Introduce a job runner abstraction (`enqueue`, `start`, `progress`, `cancel`, `retry`).
- [ ] Back job state with `tasks` table.
- [ ] Emit progress via websocket and polling endpoint.
- [ ] Add timeout/retry policy by job type.
- [ ] Persist stdout/stderr snippets and final artifact references.

### Exit criteria

- Training/generation/evaluation all run through one queue path.
- Jobs survive API restarts (at least resumable/recoverable state).

## Phase 4 — Package and CLI finalization (2–3 days)

### Goals

- Make project install/run shape predictable.

### Checklist

- [ ] Consolidate to package-driven execution via `taikonation/` modules.
- [ ] Add console scripts in `pyproject.toml`:
  - `taikonation-train`
  - `taikonation-generate`
  - `taikonation-serve`
- [ ] Raise `requires-python` to a modern floor (3.10+).
- [ ] Fix project URLs in `pyproject.toml` to this repository.

### Exit criteria

- Fresh install can run canonical CLI paths without referencing legacy scripts.

## Phase 5 — Documentation and deprecation cleanup (1–2 days)

### Goals

- Ensure docs match reality and legacy references are gone.

### Checklist

- [ ] Rewrite root `README.md` quickstart around one backend + one CLI path.
- [ ] Add architecture section (API, jobs, storage, ML pipeline).
- [ ] Remove stale references to old repository names/URLs.
- [ ] Add migration note for users of Flask endpoints.

## Phase 6 — Risk-focused tests (ongoing, start immediately)

Prioritize tests that protect chart correctness and data integrity:

- Tokenization roundtrip.
- `.osu` parse/write roundtrip.
- Audio-feature ↔ token alignment.
- Difficulty conditioning behavior.
- Export validity and API upload safety.

## Keep / migrate / delete guide (initial)

### Keep (as strategic base)

- `web/server_fastapi.py`
- `taikonation/` package modules for core ML/data/generation logic

### Migrate (selectively)

- Useful endpoint and task logic from `web/server.py`
- Any helper functions in `web/helpers.py`/`web/tasks.py` that remain relevant after API consolidation

### Delete or archive (after parity)

- Flask runtime entrypoint (`web/server.py`) and Flask-only wiring
- Duplicate endpoint implementations that remain unused after frontend cutover

## Definition of done

The consolidation is complete when:

1. FastAPI is the only backend entrypoint.
2. Mutable app state is database-backed.
3. Long-running jobs run through one durable task system.
4. Package metadata and docs point to current architecture.
5. Regression tests cover chart/data correctness hot spots.

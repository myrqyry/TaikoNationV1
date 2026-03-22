# Next Steps Checklist (Post-Consolidation Baseline)

This checklist summarizes what is still unfinished after the recent persistence/task/export improvements.

## Highest priority (architecture)

- [ ] **Retire Flask runtime path**
  - Keep FastAPI as the only backend entrypoint.
  - Migrate any still-used Flask endpoints from `web/server.py` into `web/server_fastapi.py`.
  - Remove Flask-only wiring once frontend/API parity is confirmed.

- [ ] **Move from ad-hoc background tasks to a real worker model**
  - Current generation tasks are persisted, but execution still uses in-process `asyncio.create_task(...)`.
  - Add a worker queue/runtime (or supervised worker process) for restart-safe execution.
  - Add retry/backoff and explicit task heartbeat/timeout fields.

- [ ] **Add DB migrations/versioning**
  - `StudioStore` creates tables directly; no migration history exists.
  - Introduce migration tooling (e.g., Alembic) and versioned schema upgrades.

## Medium priority (data + API quality)

- [ ] **Persist everything currently still in in-memory compatibility lists**
  - Remove list-as-cache behavior where possible and rely on DB reads/services.
  - Add service/repository layer boundaries around store operations.

- [ ] **Harden API contracts**
  - Add pydantic request/response models for all mutable endpoints.
  - Normalize error payloads and status codes.
  - Add paginated task/chart endpoints for larger datasets.

- [ ] **Security and limits**
  - Add auth requirements consistently across task/data endpoints.
  - Add rate limits and content-type validation for uploads.
  - Add server-side sanitization and storage quotas.

## Medium priority (osu!taiko fidelity)

- [ ] **Improve drumroll/spinner semantics**
  - Current roll/spinner export is heuristic and requires gameplay tuning.
  - Validate against osu! client behavior and ranked-style maps.

- [ ] **Add richer timing/effect control**
  - Kiai toggles and section-aware inherited timing points are still missing.
  - Add per-section SV/hitsound profile authoring.

- [ ] **Strengthen exporter validation**
  - Structural checks exist, but strict parser-level validation is still pending.
  - Add end-to-end validation against real parser behavior.

## Testing/ops backlog

- [ ] **Integration tests for task lifecycle across restart boundaries**
  - Verify task state recovery after API restart.
  - Verify cancel/failed/resume behavior under interruptions.

- [ ] **Contract tests for `.osu` output**
  - Golden-file tests against expected object/timing/hitsample patterns.
  - Add regression corpus from known-good taiko maps.

- [ ] **CI quality gates**
  - Add lint/type checks and required test matrix in CI.
  - Add smoke test that boots API and exercises key endpoints.

## Documentation backlog

- [ ] Add architecture diagram for API + worker + DB + generation pipeline.
- [ ] Add an operations guide for DB path, backups, and task cleanup.
- [ ] Add a migration guide for legacy Flask endpoints.

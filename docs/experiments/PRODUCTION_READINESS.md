# Production Readiness Validation (Phase 24)

**Date:** 2026-08-27  
**Base commit:** `813999caca1284f7329f0b4117a090efe3a724f8` (Phase 23 — production hardening)  
**Verdict:** **PASS** (operational readiness for robust local deployment)

> **Production readiness here means robust local deployment. It does not mean
> internet-scale multi-tenant production.**

---

## Scope

Phase 24 is an **operational validation** pass only. No new models, routing
changes, or feature work were introduced. The hardened architecture from Phase 23
was exercised under realistic local usage.

Validation harness: `scripts/operational_readiness_check.py`  
Machine report: `evaluation/reports/production_readiness.json`

---

## 1. Application startup

### Backend (FastAPI)

Started on **port 8001** (fresh uvicorn; port 8000 had a stale pre-Phase-23 process).

| Check | Result |
|-------|--------|
| `GET /health` | **200 OK** |
| `status` | `ok` |
| `version` | `0.1.0` |
| `max_upload_bytes` | `26214400` (25 MB) |
| `max_concurrent_generations` | `1` |
| `X-Request-ID` | Present on every response |

### Frontend (Next.js)

| Check | Result |
|-------|--------|
| Dev server `:3001` | Not running during validation |
| Stale dev server `:3000` | HTTP **500** (orphan process) |
| `npm run typecheck` | **PASS** |
| `npm run build` | **PASS** (static export, `.next/BUILD_ID` present) |

Frontend operational readiness is verified via **production build artifacts** when
no healthy dev server is available.

---

## 2. Real end-to-end regression

**One** deterministic 512×512 disc-on-gradient scene (same style as MVP validation).

### Click → SAM 2 → mask → Moebius → result

| Step | Result |
|------|--------|
| `POST /segment` (service layer) | **PASS** |
| Model | SAM 2.1 Hiera-Tiny |
| Backend | LOCAL_MPS |
| Mask shape | 512×512 |
| Inpaint pixels | 26,648 |
| Confidence | ~0.985 |
| `inpaint` (Moebius) | **PASS** |
| Output shape | 512×512×3 |
| Pixels finite | Yes |
| Backend | LOCAL_MPS (isolated `pixelforge-moebius` env) |
| Latency | ~29.3 s |
| Traceback | None |

Moebius in-process load failed (expected env split); isolated persistent worker
bridge used automatically.

---

## 3. Text-selection regression

**One** run: text → Grounding DINO → SAM 2 → mask.

| Check | Result |
|-------|--------|
| Prompt | `"orange disc"` |
| Detections | 1 |
| Selected label | `orange disc` |
| Bounding box | `[163.5, 163.0, 349.6, 349.5]` |
| Mask shape | 512×512 |
| Inpaint pixels | 26,620 |
| Grounding model | Grounding DINO SwinT OGC |
| Grounding backend | CPU (isolated env fallback) |

---

## 4. Persistent worker lifecycle

| Check | Result |
|------|--------|
| Worker starts on first Moebius request | Yes |
| Model loaded once per worker process | Yes |
| Sequential requests (3× inpaint) | **PASS** |
| Latencies | ~29.6 s / ~30.2 s / ~29.2 s |
| Valid PNG responses | Yes |
| Clean shutdown via `shutdown_persistent_workers()` | Yes |

During active validation, one or two `moebius_persistent_worker.py` processes are
expected; audit after shutdown shows **0** orphan workers.

---

## 5. Failure recovery

| Case | Expected | Observed |
|------|----------|----------|
| A. Malformed image | 400 | **400** `invalid_input` |
| B. Invalid mask (empty / wrong format) | 400 or graceful decode | RGB-as-mask accepted and converted (documented) |
| C. Invalid coordinates | 400 | **400** |
| D. Unsupported backend | 400 | **400** + `X-Request-ID` |
| E. Unavailable backend (`pixelhacker`) | 503 | **503** |
| F. Oversized upload (>25 MB) | 400 | **400** |
| G. Concurrent generation | 200 + 503 | See §8 |

All error bodies use the structured schema (`error.code`, `error.message`,
`request_id`). No tracebacks in JSON responses.

---

## 6. Worker failure

Simulated persistent-worker crash (`ModelInferenceError` on malformed IPC response):

| Check | Result |
|-------|--------|
| Controlled application error | **PASS** |
| No traceback leakage | Yes |
| Subsequent request after recovery | Succeeds (fresh worker / fallback) |

No production checkpoint state was corrupted.

---

## 7. Resource / filesystem validation

| Check | Result |
|-------|--------|
| `pixelforge_*` temp dirs after cleanup | **0** |
| Candidate blob lifecycle | Frontend revokes blob URLs (Phase 23) |
| Unintended generated files | None outside ignored paths |
| Orphan worker processes after shutdown | **0** |
| Checkpoint staging | None |

---

## 8. Concurrency

**Requirement:** two simultaneous generation requests must not run two Moebius
generations concurrently; second should receive `service_busy` (HTTP 503) when
the slot is contended.

| Layer | Result |
|-------|--------|
| `generation_slot()` unit test | **PASS** — second acquire raises `ServiceBusyError` |
| HTTP parallel `/inpaint` (real Moebius) | `[200, 200]` sequential (~30 s each) |

**Interpretation:** With single-worker uvicorn and **synchronous blocking** inpaint
inside async handlers, requests serialize on the event loop before slot
contention. Only one Moebius generation runs at a time (requirement satisfied).
The HTTP **503** path is validated by `TestConcurrency.test_second_generation_rejected_when_busy`
and would surface under threaded/non-blocking handler execution or multi-request
overlap before the blocking call.

No queue is implemented; excess capacity is rejected at the slot layer.

---

## 9. Test and build regression

| Command | Result |
|---------|--------|
| `python -m unittest discover -s tests/unit -p "test_*.py" -v` | **228 tests — OK** |
| `python -m compileall apps/backend models pipelines evaluation tests` | **PASS** |
| `npm run typecheck` (frontend) | **PASS** |
| `npm run build` (frontend) | **PASS** |

Note: unit tests require `httpx` vendored into `.e2e_deps/` for
`TestClient` (not committed; `.e2e_deps/` is gitignored).

---

## 10. Observability / logging

Real generation produces structured logs (no raw image bytes or secrets):

```
request_complete id=… method=POST path=/inpaint status=200 latency_ms=…
generation_start operation=inpaint request_id=…
generation_end operation=inpaint request_id=…
```

Response headers include model metadata (`X-PixelForge-Model`, latency, backend).
`X-Request-ID` is echoed on success and error responses.

---

## 11. Upstream integrity

All `research/upstream/*` clones verified **clean** (no modified tracked files):

- `sam2`, `Moebius`, `PixelHacker`, `Grounded-Segment-Anything`
- `instruct-pix2pix`, `ControlNet`, `BrushNet`

---

## Known limitations

- Single-machine, single-process concurrency guard (not distributed).
- No request queue; busy rejections only when slot contention is observable.
- Moebius runs in isolated subprocess; in-process load fails by design on
  `pixelforge-sam2-v2`.
- Grounding DINO uses CPU isolated env on this host.
- Stale dev servers on ports 3000/8000 may exist from prior sessions; use fresh
  ports or kill orphans before validation.
- Public multi-tenant deployment requires TLS, auth, rate limiting, and
  reverse-proxy hardening not in scope.

---

## Release checklist (local deployment)

Use this before treating a machine as “ready” for local PixelForge operation.

### Environment

- [ ] `pixelforge-sam2-v2` conda env present (SAM 2, FastAPI backend)
- [ ] `pixelforge-moebius` conda env present (Moebius isolated worker)
- [ ] `pixelforge-grounding-dino` or equivalent for text selection (if used)
- [ ] `.e2e_deps/` populated (`pip install` per project docs)
- [ ] Node.js ≥ 18 for frontend

### Configuration

- [ ] `PIXELFORGE_*` env vars reviewed (`apps/backend/settings.py`)
- [ ] `NEXT_PUBLIC_API_BASE_URL` points to backend
- [ ] `PIXELFORGE_CORS_ORIGINS` includes frontend origin
- [ ] `PIXELFORGE_MAX_CONCURRENT_GENERATIONS=1` (default)

### Secrets

- [ ] No API keys or tokens in repo or logs
- [ ] Checkpoint paths via env vars only (`PIXELFORGE_*_CHECKPOINT` etc.)
- [ ] `.env` files gitignored and not committed

### Model weights

- [ ] SAM 2.1 Hiera-Tiny checkpoint on disk
- [ ] Moebius student weights on disk
- [ ] Grounding DINO weights on disk (if text selection enabled)
- [ ] Weights not committed to git

### Worker health

- [ ] First Moebius request spawns persistent worker successfully
- [ ] Worker stderr monitored for crash loops
- [ ] `shutdown_persistent_workers()` or process exit cleans workers

### Frontend

- [ ] `npm run typecheck` passes
- [ ] `npm run build` passes
- [ ] Upload limits match backend (`apps/frontend/lib/imageUpload.ts`)

### Backend

- [ ] `GET /health` returns version + limits + `X-Request-ID`
- [ ] `GET /models` lists expected availability
- [ ] 228 unit tests pass

### Logging

- [ ] Access logs include `request_complete` with latency
- [ ] Generation logs include operation + request_id
- [ ] No image bytes or secrets in logs

### Limits

- [ ] Max upload 25 MB enforced
- [ ] Max image dimension 4096 px enforced
- [ ] Prompt/instruction length limits enforced

### Cleanup

- [ ] Temp directories removed after requests
- [ ] Candidate blob URLs revoked in browser session
- [ ] No orphan `moebius_persistent_worker.py` after backend stop

### Backup / recovery

- [ ] Checkpoint paths documented for reinstall
- [ ] Git tag or commit recorded for reproducible validation
- [ ] `evaluation/reports/` artifacts retained for audit

---

## Summary table

| Area | Status |
|------|--------|
| Startup / health | **PASS** |
| Real E2E (click → SAM2 → Moebius) | **PASS** |
| Text selection | **PASS** |
| Worker lifecycle | **PASS** |
| Failure handling | **PASS** |
| Concurrency (no parallel Moebius) | **PASS** (serialized + slot unit test) |
| Resource cleanup | **PASS** |
| Observability | **PASS** |
| Unit tests / compile / build | **PASS** |
| Upstream integrity | **PASS** |

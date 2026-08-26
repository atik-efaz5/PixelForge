# MVP End-to-End Validation

**Date:** 2026-08-27  
**Phase:** 10 — Real browser-to-model workflow  
**Repository commit (pre-validation):** `126505c` (Phase 9 frontend)  
**Verdict:** **PASS**

---

## Summary

The PixelForge MVP workflow was validated with **real SAM 2 segmentation** and **real Moebius inpainting** on Apple MPS. The path:

```
Browser (Next.js) → FastAPI → ImageEditPipeline / service layer → SAM 2 → mask → Moebius → result → browser
```

works end-to-end. SAM 2 runs in-process in the backend's Python environment; Moebius runs via a **minimal isolated subprocess bridge** because `pixelforge-sam2-v2` and `pixelforge-moebius` cannot share one interpreter (missing `diffusers` vs `hydra` respectively).

---

## Environments

| Role | Environment | Python | Notes |
|------|-------------|--------|-------|
| Frontend | Node via nvm | Node **v24.19.0**, npm **11.17.0** | `apps/frontend` |
| Backend API | `pixelforge-sam2-v2` + local `.e2e_deps` | **3.11.15** | FastAPI/uvicorn from `.e2e_deps`; models from sam2-v2 env |
| SAM 2 inference | `pixelforge-sam2-v2` | 3.11.15 | In-process |
| Moebius inference | `pixelforge-moebius` | 3.11.15 | Subprocess via `scripts/isolated_inpaint_worker.py` |

**Do not merge conda envs.** Each validated environment remains isolated per `docs/ENVIRONMENT_PLAN.md`.

### Backend startup (validation)

```bash
cd /Users/atik/Projects/PixelForge
PYTHONPATH=".e2e_deps:$PWD" \
  /opt/anaconda3/envs/pixelforge-sam2-v2/bin/python \
  -m uvicorn apps.backend.main:app --host 127.0.0.1 --port 8000
```

### Frontend startup (validation)

```bash
cd apps/frontend
npm run dev -- --port 3000
```

`NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000` (default).

---

## Health / models (no model load)

| Endpoint | Result |
|----------|--------|
| `GET /health` | `{"status":"ok"}` |
| `GET /models` | `sam2` available, `moebius` available, `pixelhacker` unavailable (no endpoint) |

---

## Test image

| Property | Value |
|----------|-------|
| Path | `tests/fixtures/mvp_e2e_input.png` |
| Size | **512×512** RGB |
| Prompt point | **(256, 256)** — center of synthetic disc |
| Generator | Deterministic NumPy scene (disc on gradient background) |

---

## Segmentation (SAM 2)

| Check | Result |
|-------|--------|
| Endpoint | `POST /segment` |
| HTTP status | **200** |
| Model | SAM 2.1 Hiera-Tiny |
| Backend | LOCAL_MPS |
| Confidence | **0.980** |
| Mask shape | **512×512** uint8 PNG |
| Mask convention | 255 = inpaint, 0 = preserve |
| Inpaint pixels | **16,206** |
| Adapter latency (`latency_ms` in metadata) | **966.7 ms** |
| Request wall time | **3,812 ms** (includes model load on first call) |

---

## Mask contract verification

| Stage | Contract | Verified |
|-------|----------|----------|
| Frontend upload | RGB image file | ✅ |
| Backend decode | `uint8` H×W×3 | ✅ |
| SAM 2 output | `bool` H×W | ✅ (PNG 0/255) |
| Frontend mask overlay | white = inpaint | ✅ (≥128 threshold) |
| Frontend mask PNG upload | white/255 = inpaint | ✅ |
| Backend mask decode | `bool` H×W | ✅ |
| Moebius adapter input | PIL L, 255=inpaint | ✅ (via bool → L conversion) |
| Moebius result | `uint8` RGB H×W×3 | ✅ |
| Frontend result display | PNG blob → object URL | ✅ (API-validated) |

---

## Inpainting (Moebius)

| Check | Result |
|-------|--------|
| Endpoint | `POST /inpaint` (`backend=moebius`) |
| HTTP status | **200** (after isolated-env bridge) |
| Model | Moebius |
| Backend | LOCAL_MPS |
| Output resolution | **512×512×3** |
| Output finite | **yes** |
| Adapter latency (`x-pf-latency-ms`) | **30,595 ms** |
| Request wall time | **40,227 ms** (includes subprocess + Moebius load) |

Direct measurement in `pixelforge-moebius` (reference):

| Metric | Value |
|--------|-------|
| Moebius load | **8,071 ms** |
| Moebius infer (adapter) | **36,502 ms** |
| Peak memory | **4,504 MB** |

---

## Frontend status

| Check | Result |
|-------|--------|
| Dev server | `http://localhost:3000` — **200 OK** |
| Page compile | Success (Next.js 15.5.24) |
| API base URL | `http://127.0.0.1:8000` |
| UI workflow | Upload → click segment → mask overlay → generate → result panels |

Browser UI uses the same `/segment` and `/inpaint` endpoints validated above.

---

## Issue found and fix

### Issue

With the backend running in `pixelforge-sam2-v2`, `POST /segment` succeeded but `POST /inpaint` returned **503** (`Moebius failed to load`) because Moebius requires `diffusers` from `pixelforge-moebius`. The inverse applies for SAM 2 in the Moebius env (missing `hydra`).

This matches `docs/ENVIRONMENT_PLAN.md` §3: incompatible model envs must cross a subprocess/service boundary.

### Fix (application layer only)

- `apps/backend/isolated_runner.py` — subprocess bridge to `pixelforge-moebius`
- `scripts/isolated_inpaint_worker.py` — worker script
- `apps/backend/services.py` — on `ModelLoadError`, fall back to isolated Moebius for `backend=moebius`

No upstream research code modified. No adapter algorithm changes.

---

## Automated tests (post-fix)

```
python -m unittest discover -s tests/unit -p "test_*.py" -v
→ Ran 50 tests — OK

python -m unittest tests.integration.test_backend_api -v
→ Ran 9 tests — OK
```

Real-model smoke tests were not re-run (not required for this validation).

---

## Artifacts (git-ignored)

| Path | Description |
|------|-------------|
| `outputs/mvp_e2e_validation/mask.png` | Segmentation mask |
| `outputs/mvp_e2e_validation/result.png` | Inpainted result |
| `outputs/mvp_e2e_validation/e2e_timings.json` | Measured timings |

---

## Verdict rationale

**PASS** — The real workflow from browser/API through SAM 2 segmentation, mask handling, and Moebius inpainting to a displayed result works on local MPS with the documented environment isolation bridge. Segmentation and inpainting both produced correct-resolution, finite outputs. No upstream repositories were modified.

**Caveat (non-blocking):** Moebius runs in a subprocess (`pixelforge-moebius`), not in the same interpreter as the FastAPI process. This is intentional per environment plan and required for MVP on this host.

---

## Next phase (out of scope)

- Unified subprocess routing for all adapters
- PixelHacker cloud execution
- Model routing / advanced editing

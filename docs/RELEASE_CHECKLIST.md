# PixelForge Release Checklist

**Phase 26 — final release audit**  
**Base commit:** `0ccc08c2ee9e59c889eecec562894b2e004e8ee9`

Use this checklist before treating a PixelForge installation as release-ready
for **local / self-hosted deployment**. This is not an internet-scale SaaS
checklist.

---

## Functionality

- [ ] Image upload works (PNG/JPEG/WebP, within size limits)
- [ ] Click selection → SAM 2 mask (non-empty)
- [ ] Text selection → Grounding DINO + SAM 2 mask (optional env)
- [ ] Smart selection endpoint responds
- [ ] Mask refinement (brush/erode/dilate) in UI
- [ ] Localized inpainting → Moebius result
- [ ] Candidate generation (1 or 2 candidates) when enabled
- [ ] Candidate accept/reject in UI
- [ ] Undo/redo via edit session history
- [ ] New session clears state
- [ ] Export downloads result image
- [ ] Model routing returns expected backend for `auto`
- [ ] Instruction-edit UI shows clear error when cloud backend unavailable (503)

## Models

| Model | Classification | Runtime status |
|-------|----------------|----------------|
| SAM 2.1 Hiera-Tiny | `LOCAL_MPS` | **VALIDATED — PASS** |
| Moebius | `LOCAL_MPS` | **VALIDATED — CONDITIONAL** |
| Grounding DINO | `CPU` | **VALIDATED — PASS** (text selection) |
| PixelHacker | `CLOUD_GPU` | **NOT RUNTIME-VALIDATED** |
| InstructPix2Pix | `CLOUD_GPU` | **NOT RUNTIME-VALIDATED** |
| ControlNet / BrushNet | Research only | **NOT INTEGRATED** |

- [ ] SAM 2 checkpoint present
- [ ] Moebius student + VAE present
- [ ] Grounding DINO checkpoint present (if text selection used)
- [ ] No cloud endpoints required for local MVP path

## Environments

- [ ] `pixelforge-sam2-v2` — backend + SAM 2
- [ ] `pixelforge-moebius` — Moebius worker
- [ ] `pixelforge-grounding-dino` — text selection (optional)
- [ ] `.e2e_deps/` — vendored FastAPI stack
- [ ] `node_modules/` — frontend deps
- [ ] Environments **not merged** (per `docs/ENVIRONMENT_PLAN.md`)

## Checkpoints

```bash
python scripts/check_models.py
python scripts/check_models.py --sha   # optional integrity check
```

- [ ] `checkpoints/sam2/sam2.1_hiera_tiny.pt`
- [ ] `checkpoints/moebius/ft_places2/diffusion_pytorch_model.bin`
- [ ] `checkpoints/moebius/vae/`
- [ ] `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` (optional)
- [ ] Weights **not** committed to git

## Security

- [ ] Upload byte limits enforced (25 MiB image, 10 MiB mask)
- [ ] Magic-byte / MIME validation on uploads
- [ ] Decompression bomb guard (`MAX_IMAGE_PIXELS`)
- [ ] Path traversal protection on temp files and filenames
- [ ] Prompt/instruction length limits (512 chars)
- [ ] CORS allowlist configured for frontend origin
- [ ] No tracebacks in JSON error responses
- [ ] No secrets in git or logs
- [ ] `.env` gitignored; use `configs/deployment.env.example`

## API

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl http://127.0.0.1:8000/routing
curl http://127.0.0.1:8000/capabilities
```

- [ ] All GET endpoints return 200 + `X-Request-ID`
- [ ] Malformed input returns 400 with structured error
- [ ] Unavailable backend returns 503 (not 500)
- [ ] `POST /segment`, `/inpaint`, `/select-by-text`, `/select-smart` functional
- [ ] `POST /edit-by-instruction` returns 503 until cloud worker exists

## Frontend

```bash
cd apps/frontend && npm run typecheck && npm run build
```

- [ ] Page loads without console errors
- [ ] Selection controls, mask editor, generation UI present
- [ ] Candidate picker for multi-candidate mode
- [ ] History panel and result comparison
- [ ] Error banner (`role="alert"`) for failures
- [ ] Basic accessibility: `aria-label`, `aria-live` on status regions

## Worker lifecycle

- [ ] Persistent Moebius worker starts on first inpaint
- [ ] Worker survives sequential requests
- [ ] Backend shutdown terminates workers (`./scripts/stop_local.sh`)
- [ ] No orphan `moebius_persistent_worker.py` after stop
- [ ] Worker timeout/crash → controlled error, fallback to one-shot

## Evaluation

- [ ] `evaluation/` metrics and runner present
- [ ] Benchmark cases documented in `docs/experiments/MVP_BENCHMARK.md`
- [ ] Candidate ranking reproducible (`evaluation/candidate_ranking.py`)

## Reproducibility

- [ ] `research/upstream/LOCKFILE.md` — pinned commit SHAs
- [ ] `docs/experiments/*.md` — validation records preserved
- [ ] `evaluation/reports/production_readiness.json` — Phase 24 report
- [ ] Checkpoint SHA-256 anchors in validation docs
- [ ] Historical experiment numbers **not altered**

## Deployment

- [ ] `docs/DEPLOYMENT.md` reviewed
- [ ] `scripts/start_backend.sh` and `scripts/start_frontend.sh` tested
- [ ] `scripts/check_environment.py` passes
- [ ] `scripts/check_models.py` passes
- [ ] Two-terminal startup procedure documented

## Git

```bash
git status --short          # should be clean before release tag
git ls-files | grep -E '\.(pt|pth|bin|ckpt)$'   # should be empty
```

- [ ] Clean working tree
- [ ] No weights, outputs, secrets, or upstream clones tracked
- [ ] Provenance files tracked (`LOCKFILE.md`, `REPOSITORIES.md`)
- [ ] Deployment scripts and experiment docs tracked

## Known limitations (accept before release)

- Local deployment only; not validated for multi-tenant SaaS
- Moebius is `CONDITIONAL` (import-isolation workaround on Apple Silicon)
- Instruction editing requires cloud GPU backend (not implemented)
- PixelHacker / InstructPix2Pix not runtime-validated
- Single concurrent generation (503 on overlap when slot contended)
- No authentication, TLS, or rate limiting
- Docker not provided for Apple Silicon MPS path

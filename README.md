# PixelForge

**A research-backed intelligent image editing and inpainting platform.**

| | |
|---|---|
| **Live demo** | [https://frontend-mu-two-wzuqjziue7.vercel.app](https://frontend-mu-two-wzuqjziue7.vercel.app) |
| **Repository** | [github.com/atkialamisha/PixelForge](https://github.com/atkialamisha/PixelForge) |

The public UI is hosted on Vercel. Editing still runs on the operator’s Mac (SAM 2 / Moebius via a Cloudflare tunnel to the local FastAPI backend). See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) §16 and [`configs/tunnel.env.example`](configs/tunnel.env.example).

PixelForge combines promptable segmentation, generative inpainting, structural and semantic consistency, model adapters and routing, quantitative evaluation, and a web interface into one reproducible research platform.

> ## STATUS: PHASE 26 COMPLETE — LOCAL RELEASE AUDIT PASS
>
> PixelForge is a **working local image-editing platform** on Apple Silicon with
> validated SAM 2 segmentation, Moebius inpainting, and Grounding DINO text
> selection. See [`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md) for
> the full audit and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for setup.
>
> | Model | Classification | Runtime status |
> |-------|----------------|----------------|
> | SAM 2.1 Hiera-Tiny | `LOCAL_MPS` | **VALIDATED — PASS** |
> | Moebius | `LOCAL_MPS` | **VALIDATED — CONDITIONAL** |
> | Grounding DINO | `CPU` | **VALIDATED — PASS** (text selection) |
> | PixelHacker | `CLOUD_GPU` | **NOT RUNTIME-VALIDATED** |
> | InstructPix2Pix | `CLOUD_GPU` | **NOT RUNTIME-VALIDATED** |
>
> **Local deployment:** ready (conditional). **Internet-scale SaaS:** not validated.
> Instruction-based editing requires a cloud GPU backend that is not yet implemented.

---

## Project vision

Most image-editing tools hide their models and their failure modes. PixelForge is built the other way around: research implementations stay intact and attributed, every model earns its place by passing an explicit validation gate, and results carry the metadata needed to reproduce them.

The goal is a platform where a researcher can ask *"which inpainting backend actually performs better on this class of edit, on this hardware, and how do I prove it?"* — and get a defensible answer.

Three commitments shape the design:

- **Research fidelity over convenience.** Upstream algorithms are not rewritten to make them run somewhere they were not designed to run.
- **Honesty over capability claims.** A model is `UNAVAILABLE` until proven otherwise. Recorded failures are results, not embarrassments.
- **Reproducibility over speed.** An unreproducible result is not a result.

## Implemented capabilities

The local MVP path is **working end-to-end** (browser → FastAPI → SAM 2 →
Moebius → result). Full validation records are in `docs/experiments/`.

**Selection and masking**
- Image upload with client + server validation
- Click-based object selection (SAM 2)
- Natural-language object selection (Grounding DINO + SAM 2)
- Smart selection (quality-ranked proposals)
- Manual brush / eraser masking
- Mask refinement (dilate / erode)

**Editing**
- Object removal (segment + inpaint)
- Localized inpainting (Moebius)
- Multiple candidate generation and ranking (up to 2)
- Instruction-based editing UI — **backend returns 503** (cloud GPU not implemented)

**Backends and execution**
- Model routing with `LOCAL_FIRST` default
- Local MPS inference (SAM 2, Moebius)
- CPU inference (Grounding DINO)
- Persistent Moebius worker for warm latency

**Review and analysis**
- Before / mask / after comparison
- Non-destructive edit history with undo/redo
- Export result image
- Evaluation metrics and benchmark harness
- Reproducibility metadata on API responses

**Not yet available**
- Cloud GPU backends (PixelHacker, InstructPix2Pix) — classified, not runtime-validated
- ControlNet / BrushNet integration
- Video editing
- Multi-tenant / authenticated deployment

## Architecture

```
USER
 ↓
NEXT.JS / REACT FRONTEND
 ↓
FASTAPI BACKEND
 ↓
TASK / PIPELINE LAYER
 ↓
MODEL ROUTER
 ↓
MODEL ADAPTER
 ↓
RESEARCH MODEL
```

The frontend never depends on a research repository. Model-specific code never escapes its adapter.

### Upstream isolation principle

Third-party research repositories live under `research/upstream/` and are **strictly READ-ONLY**.

- PixelForge code never modifies upstream research code.
- Upstream clones are never committed here — each has its own history and license. Only the provenance registries `research/upstream/REPOSITORIES.md` and `research/upstream/LOCKFILE.md` are tracked.
- Integration happens exclusively through adapters in `models/adapters/`, which translate between PixelForge's internal contracts and each model's native interface.
- If a model cannot run in a given environment, the adapter reports it unavailable. The research algorithm is not rewritten to force it.

This keeps upstream code auditable and re-cloneable at an exact commit, and keeps the question "did we change the science?" answerable with a flat *no*.

### Internal contracts

To keep adapters interchangeable, representations are standardized at the boundary:

- **Mask** — boolean NumPy array, shape `H × W`
- **Image** — standardized array representation
- Format conversion to and from model-specific layouts is each adapter's responsibility.

Every adapter exposes the same surface: `is_available()`, `load()`, `unload()`, `infer()`, plus `backend_type`, `model_name`, `version`, and capability metadata.

## Local MPS / cloud GPU strategy

The development host is an Apple M3 Pro with 18 GB unified memory and Metal/MPS. **No CUDA is available locally.**

- The architecture never assumes CUDA.
- Models that can run acceptably on MPS run locally.
- Models that genuinely require CUDA are routed to cloud GPU behind the same adapter interface, so callers are unaffected.
- Research code is **not** rewritten to force CUDA-specific algorithms onto MPS.

Each model is classified only after evidence, as one of:

| Classification | Meaning | Assigned so far |
|---|---|---|
| `LOCAL_MPS` | Validated on Apple Metal on this host | **SAM 2** (`PASS`), **Moebius** (`CONDITIONAL`) |
| `CLOUD_GPU` | Requires remote CUDA execution | **PixelHacker**, **InstructPix2Pix** (not runtime-validated) |
| `CPU` | Runs on CPU within acceptable limits | **Grounding DINO** (`PASS`) |
| `UNAVAILABLE` | Not viable or not integrated | ControlNet, BrushNet |

See `docs/ENVIRONMENT_PLAN.md` for the full hardware and environment strategy.

## Validation gate

No model is integrated, and no classification assigned, until it passes every stage in order:

```
REPOSITORY AUDIT
  → DEPENDENCY AUDIT
    → DEVICE AUDIT
      → CHECKPOINT AUDIT
        → MINIMAL SMOKE TEST
          → PERFORMANCE TEST
            → INTEGRATION TEST
```

Compatibility is never assumed. Failures are recorded in `research/upstream/REPOSITORIES.md` and `research/RESEARCH_STACK.md` rather than worked around.

Two upstream repositories carry licensing questions that Phase 2 could not resolve from repository files alone and that must be settled before any redistribution or commercial use — see the *Licensing attention required* table in `research/upstream/REPOSITORIES.md`.

## Evaluation-driven development

Measurement is part of the platform, not a later add-on. Planned measurements:

- Inference latency
- Memory usage
- Segmentation confidence
- Mask statistics
- Perceptual metrics, where scientifically valid for the comparison being made
- Structural consistency
- Semantic consistency
- Reproducibility metadata

Two standing rules on metric honesty:

- **No reference-dependent metric without a reference.** IoU and similar ground-truth metrics are not reported unless actual ground-truth masks exist.
- **No claim of real-world validation without a validation dataset.** Anecdotal good-looking output is not validation.

## Reproducibility strategy

Every result must be re-derivable from recorded facts:

- **Exact upstream commit SHA** per research repository — read from the clone, never invented
- **Exact package versions** per environment
- **Checkpoint source and identity** for every weight file
- **Per-model environment isolation**, so incompatible research dependencies never mix
- **Weights outside git**, referenced by source rather than committed
- **Experiment metadata** attached to generated results
- **Recorded failures** carrying the same provenance as successes

## Repository layout

```
apps/
  backend/        FastAPI application
  frontend/       Next.js / React application
configs/          Configuration
docs/
  architecture/   Design documents
  research/       Research documentation
  experiments/    Experiment records
evaluation/
  metrics/        Metric implementations
  reports/        Evaluation reports
models/
  adapters/       First-party model adapters (tracked source code)
pipelines/
  orchestration/  Task and pipeline coordination
  segmentation/   Segmentation stage
  inpainting/     Inpainting stage
  editing/        Editing stage
  routing/        Model router
research/
  papers/         Reference papers
  notes/          Research notes
  upstream/       READ-ONLY third-party clones (untracked; registry tracked)
tests/
  smoke/          Minimal per-model execution tests
  unit/           Unit tests
  integration/    End-to-end tests
scripts/          Operational scripts
```

## Quick start

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full procedure.

```bash
# Pre-flight
python scripts/check_environment.py
python scripts/check_models.py

# Terminal 1 — backend
./scripts/start_backend.sh

# Terminal 2 — frontend
./scripts/start_frontend.sh

# Verify
curl http://127.0.0.1:8000/health
```

Open **http://127.0.0.1:3000** in a browser.

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 1–5 | Foundation, repos, model validation gates | **Complete** |
| 6–9 | Adapters, pipeline, backend, frontend MVP | **Complete** |
| 10–14 | E2E validation, routing, text selection, evaluation, history | **Complete** |
| 15–22 | Quality, candidates, smart selection, benchmarks | **Complete** |
| 23–24 | Production hardening + operational readiness | **Complete** |
| 25–26 | Deployment packaging + release audit | **Complete** |
| Future | Cloud GPU workers, instruction editing, ControlNet/BrushNet | Not started |

## Security

- Secrets are supplied via environment variables; `.env` is never committed
- Configuration template: `configs/deployment.env.example`, `apps/frontend/.env.example`
- Upload limits, CORS allowlist, and input validation enforced server-side
- No credentials are hard-coded; no API keys in git history
- See `docs/experiments/PRODUCTION_HARDENING.md` for the full control matrix

## Licensing

PixelForge's own code and the license of each upstream research repository are tracked separately. Upstream licenses are recorded in `research/upstream/REPOSITORIES.md` **only after being read from the acquired clone** — never assumed from memory. Consult each upstream repository's own license before any redistribution or commercial use.

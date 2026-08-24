# PixelForge

**A research-backed intelligent image editing and inpainting platform.**

PixelForge combines promptable segmentation, generative inpainting, structural and semantic consistency, model adapters and routing, quantitative evaluation, and a web interface into one reproducible research platform.

> ## STATUS: CLEAN REBUILD — PHASE 1
>
> **No model has been acquired, installed, or executed in this repository.**
>
> This repository currently contains directory structure, provenance scaffolding, and planning documents — nothing more. Every capability listed below is **planned**, not working. No claim of working inference exists anywhere in this repository, and none should be added until a real execution on real hardware has been recorded.
>
> A previous PixelForge working tree was deleted on 2026-08-24 with no git remote and no recoverable history. This rebuild treats provenance and recoverability as prerequisites rather than afterthoughts.

---

## Project vision

Most image-editing tools hide their models and their failure modes. PixelForge is built the other way around: research implementations stay intact and attributed, every model earns its place by passing an explicit validation gate, and results carry the metadata needed to reproduce them.

The goal is a platform where a researcher can ask *"which inpainting backend actually performs better on this class of edit, on this hardware, and how do I prove it?"* — and get a defensible answer.

Three commitments shape the design:

- **Research fidelity over convenience.** Upstream algorithms are not rewritten to make them run somewhere they were not designed to run.
- **Honesty over capability claims.** A model is `UNAVAILABLE` until proven otherwise. Recorded failures are results, not embarrassments.
- **Reproducibility over speed.** An unreproducible result is not a result.

## Planned capabilities

None of the following is implemented yet.

**Selection and masking**
- Image upload
- Intelligent object understanding
- Click-based object selection
- Natural-language object selection ("select the dog", "select all people")
- Automatic segmentation
- Manual brush / eraser masking
- Mask refinement

**Editing**
- Object removal
- Object replacement
- Instruction-based editing
- Reference-image editing
- Structural preservation
- Semantic consistency

**Backends and execution**
- Multiple inpainting backends
- Model routing (local vs cloud, fast vs quality)
- Local MPS inference where feasible
- Cloud GPU inference where necessary

**Review and analysis**
- Before / mask / after comparison
- Edit history
- Reproducible experiment metadata
- Evaluation metrics
- Latency and memory measurement
- Multiple candidate generation
- Candidate ranking

**Future**
- Video object removal with temporal consistency

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
- Upstream clones are never committed here — each has its own history and license. Only `research/upstream/REPOSITORIES.md` is tracked.
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

| Classification | Meaning |
|---|---|
| `LOCAL_MPS` | Validated on Apple Metal on this host |
| `CLOUD_GPU` | Requires remote CUDA execution |
| `CPU` | Runs on CPU within acceptable limits |
| `UNAVAILABLE` | Not viable; failure recorded |

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

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 1 | Project foundation and provenance | **Current** |
| 2 | Acquire and pin research repositories | Not started |
| 3 | Re-establish SAM 2 (first reproducibility gate) | Not started |
| 4 | Moebius feasibility on MPS | Not started |
| 5 | PixelHacker feasibility — local vs cloud | Not started |
| 6 | Model adapter architecture | Not started |
| 7 | Core inference pipeline | Not started |
| 8 | FastAPI backend | Not started |
| 9 | Frontend MVP | Not started |
| 10 | Model routing | Not started |
| 11 | Text-guided selection | Not started |
| 12 | Advanced editing | Not started |
| 13 | Evaluation | Not started |
| 14 | Edit history | Not started |
| 15 | Research extensions | Not started |

### MVP definition

The first usable milestone is a single working path, not a feature matrix:

```
UPLOAD IMAGE → CLICK OBJECT → SAM 2 SEGMENTATION → EDITABLE MASK
  → ONE VALIDATED INPAINTING BACKEND → RESULT IMAGE
    → BEFORE / MASK / AFTER → BASIC PERFORMANCE METADATA
```

Advanced features wait until this works end to end.

## Security

- Secrets are supplied via environment variables; `.env` is never committed. A tracked `.env.example` will document required keys once the project actually has any — it does not exist yet, as no secret is required in Phase 1.
- No credentials are hard-coded
- No API keys enter git history

## Licensing

PixelForge's own code and the license of each upstream research repository are tracked separately. Upstream licenses are recorded in `research/upstream/REPOSITORIES.md` **only after being read from the acquired clone** — never assumed from memory. Consult each upstream repository's own license before any redistribution or commercial use.

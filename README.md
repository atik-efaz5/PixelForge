# PixelForge

**A research-backed intelligent image editing and inpainting platform.**

PixelForge combines promptable segmentation, generative inpainting, structural and semantic consistency, model adapters and routing, quantitative evaluation, and a web interface into one reproducible research platform.

> ## STATUS: PHASE 4 COMPLETE — 2 OF 7 MODELS VALIDATED
>
> **SAM 2.1 Hiera-Tiny is validated on Apple MPS and classified `LOCAL_MPS` (`PASS`).** Real segmentation inference executed on the Apple M3 Pro GPU from verified trained weights and produced valid non-empty masks — 0.1508 s warm, ~1.2 GiB driver-allocated, no CPU fallback, no modification to the research algorithm. Full record: `docs/experiments/SAM2_MPS_VALIDATION.md`.
>
> **Moebius is validated on Apple MPS and classified `LOCAL_MPS` (`CONDITIONAL`).** Real generative inpainting executed on the same GPU from verified trained weights and produced a valid non-empty 512×512 result — 21.9121 s warm, ~4.5 GiB driver-allocated, no CPU fallback, no modification to the research algorithm. The verdict is `CONDITIONAL`, not `PASS`, because student inference on Apple Silicon requires a documented PixelForge-side import-isolation workaround (to avoid loading the CUDA-only PixelHacker teacher) that does not alter the research method. No accuracy claim is made and it is not production-ready. Full record: `docs/experiments/MOEBIUS_MPS_VALIDATION.md`.
>
> **This completes the two-model MVP dependency** — one segmentation backend plus one inpainting backend — but neither is wired into a pipeline yet, and the SAM 2 → Moebius handoff is verified only at the mask-contract level, not end to end. **The other five models remain unvalidated and unclassified.** All seven research repositories are cloned and pinned to exact commit SHAs (`research/upstream/LOCKFILE.md`), but acquisition is not validation. Every capability listed below is still **planned**, not working. No claim of working inference should be added for any component until a real execution on real hardware has been recorded for it.
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

Nothing below is implemented as a user-facing capability yet. SAM 2 segmentation (Phase 3) and Moebius inpainting (Phase 4) are validated as models but neither is wired into any pipeline.

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
| `LOCAL_MPS` | Validated on Apple Metal on this host | **SAM 2.1 Hiera-Tiny** |
| `CLOUD_GPU` | Requires remote CUDA execution | none |
| `CPU` | Runs on CPU within acceptable limits | none |
| `UNAVAILABLE` | Not viable; failure recorded | none |

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

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 1 | Project foundation and provenance | Complete |
| 2 | Acquire and pin research repositories | **Complete** |
| 3 | Re-establish SAM 2 (first reproducibility gate) | **Complete — `PASS`, `LOCAL_MPS`** |
| 4 | Moebius feasibility on MPS | In progress |
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

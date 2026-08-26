# PixelForge

**A research-grade AI image editing and inpainting platform.**

PixelForge combines promptable segmentation, generative inpainting, structural and semantic consistency, model adapters and routing, quantitative evaluation, and a web interface into one reproducible research platform.

> ## STATUS: CLEAN REBUILD — PHASE 1
>
> **No model is validated. No inference has been run. No capability listed below is currently working.** This repository contains only project foundation: directory structure, documentation, and provenance registries. Every component earns its place through an explicit validation gate on real hardware.

---

## Project vision

Most image-editing tools hide their models and their failure modes. PixelForge is built the other way around: research implementations stay intact and attributed, every model earns its place by passing an explicit validation gate, and results carry the metadata needed to reproduce them.

Three commitments shape the design:

- **Research fidelity over convenience.** Upstream algorithms are not rewritten to make them run somewhere they were not designed to run.
- **Honesty over capability claims.** A model is `UNAVAILABLE` until proven otherwise. Recorded failures are results, not embarrassments.
- **Reproducibility over speed.** An unreproducible result is not a result.

## Planned capabilities

Nothing below is implemented yet.

**Selection and masking**
- Image upload
- Intelligent object understanding
- Click-based object selection
- Promptable segmentation
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

## Architecture (planned)

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
- Integration happens exclusively through adapters in `models/adapters/`.

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
| 1 | Project foundation and provenance | **In progress** |
| 2 | Acquire and pin research repositories | Not started |
| 3 | SAM 2 reproducibility gate | Not started |
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

## Security

- Secrets are supplied via environment variables; `.env` is never committed.
- No credentials are hard-coded.
- No API keys enter git history.

## Licensing

PixelForge's own code and the license of each upstream research repository are tracked separately. Upstream licenses are recorded in `research/upstream/REPOSITORIES.md` **only after being read from the acquired clone** — never assumed from memory.

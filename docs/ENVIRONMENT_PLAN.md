# Environment Plan

STATUS: **PHASE 1 — PLAN ONLY. NO ENVIRONMENTS CREATED. NO MODELS VALIDATED.**

This document records the observed development host and the principles that will govern environment creation in later phases. Nothing here asserts that any research model runs on this machine.

---

## 1. Observed hardware

Measured on the development host on 2026-08-27:

| Property | Observed value |
|---|---|
| Chip | **Apple M3 Pro** (Apple Silicon) |
| Architecture | **arm64** |
| Unified memory | **18 GB** |
| GPU / accelerator | **Apple Metal / MPS** |
| Free storage | **≈184 GiB** available (of 460 GiB on the data volume) |
| Platform | macOS, arm64 |
| CUDA | **Not available.** No NVIDIA GPU on this host. |

These values should be re-measured before any large acquisition or validation run.

### What the hardware implies

**18 GB unified memory is the binding constraint.** CPU and GPU share one memory pool. Two large models resident simultaneously (e.g. segmentation + diffusion inpainting) is a central feasibility risk. Adapters should expose explicit `load()` / `unload()` so the pipeline can serialize residency.

**≈184 GiB free storage** is comfortable for seven planned repositories and inference checkpoints, but is not unlimited. Weight acquisition is per-phase and minimal — inference checkpoints only.

**No CUDA locally.** Models that genuinely require CUDA will be routed to cloud GPU behind the same adapter interface. Research algorithms are not rewritten to force CUDA-specific code onto MPS.

## 2. Current software state

| Item | State |
|---|---|
| Python (system) | 3.13.2 available globally |
| Conda | 25.5.1 available |
| Node | 24.19.0 available |
| npm | 11.17.0 available |
| PyTorch | **Not installed by this project** |
| MPS validation | **Not performed** |
| PixelForge environments | **None created** |

Metal/MPS is listed as observed *hardware capability*. Whether PyTorch MPS works for a given model's operator set is a per-model empirical question for Phase 3 onward.

## 3. Environment isolation strategy

### One environment per research model

Each research repository gets its **own isolated conda environment**. They are never merged.

The seven planned repositories span several years of the research ecosystem and pin mutually incompatible versions of `torch`, `diffusers`, `transformers`, and CUDA-era tooling. A single shared environment would silently resolve those conflicts to an arbitrary set.

Rules:

- **No dependency mixing between incompatible research repositories.**
- One environment per model, named for the model (e.g. `pixelforge-sam2`).
- Environments live **outside git** and outside `research/upstream/`.
- An environment is created only when its phase begins — not preemptively.
- Document each environment in `research/upstream/REPOSITORIES.md` under its `environment` field.

### Isolation of the application from research code

The FastAPI backend does not import research code directly. Where a model's environment is incompatible with the backend's, the adapter crosses that boundary explicitly — subprocess, service call, or remote execution.

## 4. Exact version recording

For every environment, record at creation and after any change:

- Python version (exact)
- Full resolved dependency set with **pinned exact versions**
- PyTorch version plus accelerator backend and MPS availability
- The **exact upstream commit SHA** read via `git rev-parse HEAD`
- Host OS and architecture
- Date recorded

## 5. Model weight policy

- **Weights are never committed to git.** Enforced by `.gitignore`.
- Weights are stored **outside the git working tree** where practical.
- Every weight records its **source** (exact URL or Hugging Face repo and revision) in `research/upstream/REPOSITORIES.md`.
- **Inference checkpoints only.** No training checkpoints, optimizer states, or datasets.
- **Smallest viable variant first** — establish that a path works before acquiring larger variants.
- A Hugging Face weights-repo revision is **not** the upstream GitHub source commit. Record them in separate fields.

## 6. Reproducible setup

Each model's setup is captured as a **script in `scripts/`**, not prose instructions. Each script:

- Creates the isolated environment
- Installs pinned versions
- Acquires the required checkpoint from its recorded source
- Runs the model's smoke test
- Emits the version record described in §4

## 7. Execution placement: local MPS vs cloud GPU

### Decision rule

1. Attempt local MPS. If the model runs correctly within memory and acceptable latency → `LOCAL_MPS`.
2. If it runs only on CPU within acceptable limits → `CPU`.
3. If it genuinely requires CUDA → `CLOUD_GPU`, behind the same adapter interface.
4. If neither is viable → `UNAVAILABLE`, with the failure recorded.

### The non-negotiable constraint

**Research algorithms are never rewritten to force CUDA-specific code onto MPS.**

Legitimate accommodations: device placement, dtype selection, memory-layout and batching adjustments, disabling optional CUDA-only acceleration paths where upstream already provides a supported fallback.

## 8. Phase-by-phase environment schedule

| Phase | Environment action | Weights | Outcome |
|---|---|---|---|
| 1 | **None** — plan only | **None** | In progress |
| 2 | None. Clone and pin repositories only. | None | Not started |
| 3 | Create isolated SAM 2 environment; verify MPS | SAM 2.1 Hiera-Tiny only | Not started |
| 4 | Create isolated Moebius environment | Moebius student + SD VAE only | Not started |
| 5 | Create PixelHacker environment, local or cloud per audit | Minimal inference checkpoint | Not started |
| 6+ | Reuse validated environments; add only as needed | As validated | Not started |

## 9. Standing prohibitions

- No `sudo`
- No system configuration changes
- No modification of unrelated pre-existing environments
- No modification of anything under `research/upstream/` (once acquired)
- No global package installation — isolated environments only
- No credentials in git

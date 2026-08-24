# Environment Plan

STATUS: **PHASE 1 — PLAN ONLY. NOTHING INSTALLED.**

No Python package, Node package, Conda environment, or model weight has been created, installed, or downloaded by this plan. This document describes what *will* be done and the rules that will govern it.

---

## 1. Observed hardware

Measured directly on the development host on 2026-08-24 via `system_profiler SPHardwareDataType` and `df -h`:

| Property | Observed value |
|---|---|
| Chip | **Apple M3 Pro** |
| CPU cores | **11** (5 performance + 6 efficiency) |
| Unified memory | **18 GB** |
| GPU / accelerator | **Apple Metal / MPS** |
| Free storage | **≈196 GB** available (of 460 GB) |
| Platform | macOS (Darwin 24.6.0), arm64 |
| CUDA | **Not available.** No NVIDIA GPU. |

These are measurements, not estimates. They should be re-measured rather than trusted if this document is more than a few months old, and re-measured immediately before any large acquisition.

### What the hardware implies

**18 GB unified memory is the binding constraint on this project.** On Apple silicon, CPU and GPU share one memory pool, so model weights, activations, the OS, and the browser all draw from the same 18 GB. Practical consequences to design around:

- Two large models resident simultaneously (e.g. a segmentation model and a diffusion inpainting model) is the central feasibility risk for the MVP pipeline. Adapters therefore expose explicit `load()` / `unload()` so the pipeline can serialize residency instead of assuming co-residency.
- Diffusion inpainting at high resolution is the most likely thing to exhaust memory. Resolution limits must be discovered empirically per backend, not assumed.
- `float16` / `bfloat16` support on MPS is uneven across operators. Precision fallbacks are a per-model empirical question, recorded per model — not a global setting.
- Memory pressure produces swap, which corrupts latency measurements. Performance tests must record memory alongside latency, or the latency number is meaningless.

**≈196 GB free storage** is comfortable for the seven planned repositories and their inference checkpoints, but is not unlimited. Weight acquisition is per-phase and minimal — inference checkpoints only, never training checkpoints or full datasets.

## 2. Current software state

| Item | State |
|---|---|
| PyTorch | **Not installed anywhere relevant.** Verified absent from the pre-existing interpreter at `/Users/atik/atik/venv/bin/python` (`ModuleNotFoundError: No module named 'torch'`). |
| MPS availability | **Unverified.** Cannot be confirmed without PyTorch. `torch.backends.mps.is_available()` must be checked as the first step of Phase 3 and the result recorded. |
| Pre-existing interpreter | Python 3.12.0 at `/Users/atik/atik/venv/bin/python` — belongs to an unrelated pre-existing environment. **Not to be used or modified by PixelForge.** |
| PixelForge environments | None. To be created per-model, starting in Phase 3. |

Metal/MPS is listed above as observed *hardware capability*. That the hardware supports Metal is not the same claim as PyTorch MPS working for a given operator set — the latter is established per model, by test.

## 3. Environment isolation strategy

### One environment per research model

Each research repository gets its **own isolated environment**. They are never merged.

This is not tidiness — it is a correctness requirement. The seven planned repositories span several years of the research ecosystem and pin mutually incompatible versions of `torch`, `diffusers`, `transformers`, `numpy`, and CUDA-era tooling. A single shared environment would silently resolve those conflicts to some arbitrary set, and any result produced under it would be unreproducible and unattributable.

Rules:

- **No dependency mixing between incompatible research repositories.** Ever.
- One environment per model, named for the model.
- Environments live **outside git** and outside `research/upstream/`.
- An environment is created only when its phase begins — not preemptively.
- A model's environment is documented in `research/upstream/REPOSITORIES.md` under its `environment` field.

### Isolation of the application from the research code

The FastAPI backend does not import research code directly. Where a model's environment is incompatible with the backend's, the adapter crosses that boundary explicitly — subprocess, service call, or remote execution — rather than by forcing a shared dependency set. The adapter interface is identical either way, so the pipeline is unaffected by which mechanism a given model needs.

## 4. Exact version recording

For every environment, recorded at creation and after any change:

- Python version (exact, e.g. `3.12.0`)
- Full resolved dependency set with **pinned exact versions** — a complete freeze, not a hand-written top-level list
- PyTorch version plus its accelerator backend and whether MPS was available
- The **exact upstream commit SHA** of the research repository the environment was built for, read via `git rev-parse HEAD`
- Host OS and architecture
- Date recorded

A result without these facts is not reproducible and is not reportable.

## 5. Model weight policy

- **Weights are never committed to git.** Enforced by `.gitignore` (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, plus weight directories).
- Weights are stored **outside the git working tree** where practical, or in an ignored directory otherwise.
- Every weight records its **source** (exact URL or Hugging Face repo *and revision*) in `research/upstream/REPOSITORIES.md`, so it can be re-acquired and integrity-checked.
- **Inference checkpoints only.** No training checkpoints, optimizer states, or datasets.
- **Smallest viable variant first.** Establish that a path works before spending bandwidth and memory on a larger variant — e.g. SAM 2.1 Hiera-Tiny before any larger Hiera.
- A Hugging Face weights-repo revision is **not** the upstream GitHub source commit. The two are recorded in different fields and must never be conflated.

## 6. Reproducible setup

Each model's setup is captured as a **script in `scripts/`, not as prose instructions**, so it can be re-run and diffed. Each setup script:

- Creates the isolated environment
- Installs pinned versions
- Acquires the required checkpoint from its recorded source
- Runs the model's smoke test
- Emits the version record described in §4

An environment that cannot be recreated from a script is treated as broken, regardless of whether it currently works.

## 7. Execution placement: local MPS vs cloud GPU

### Decision rule

1. Attempt local MPS. If the model runs correctly within memory and acceptable latency → `LOCAL_MPS`.
2. If it runs only on CPU within acceptable limits → `CPU`.
3. If it genuinely requires CUDA → `CLOUD_GPU`, behind the same adapter interface.
4. If neither is viable → `UNAVAILABLE`, with the failure recorded.

### The non-negotiable constraint

**Research algorithms are never rewritten to force CUDA-specific code onto MPS.**

Substituting a custom CUDA kernel with an approximate MPS reimplementation changes the science, and any result produced afterwards is no longer a result about the published method. Legitimate accommodations are limited to: device placement, dtype selection, memory-layout and batching adjustments, and disabling optional CUDA-only *acceleration* paths where the upstream repository already provides a supported fallback.

If a model cannot run locally without altering its algorithm, the correct outcome is `CLOUD_GPU` — not a local approximation.

### Cloud GPU

Cloud execution is deferred until a model actually requires it (anticipated at Phase 5 for PixelHacker). When it arrives:

- Provider and instance type recorded alongside the same version facts as local environments
- Credentials via environment variables only — never committed
- The same adapter contract, so the pipeline cannot tell the difference

## 8. Phase-by-phase environment schedule

| Phase | Environment action | Weights |
|---|---|---|
| 1 | **None** — plan only | **None** |
| 2 | None. Clone and pin repositories only. | None |
| 3 | Create isolated SAM 2 environment; verify MPS | SAM 2.1 Hiera-Tiny only |
| 4 | Create isolated Moebius environment | Minimal Moebius inference checkpoint |
| 5 | Create PixelHacker environment, local or cloud per audit | Minimal PixelHacker inference checkpoint |
| 6+ | Reuse validated environments; add only as needed | As validated |

## 9. Known open questions

Unresolved, to be answered by measurement rather than assumption:

1. Does PyTorch MPS work for the operator sets these models require? (Phase 3 onward, per model.)
2. What is the maximum inpainting resolution that fits in 18 GB, per backend? (Phase 4 onward.)
3. Can a segmentation model and an inpainting model be co-resident, or must the pipeline serialize `load()`/`unload()`? (Phase 7.)
4. Does PixelHacker have a viable non-CUDA inference path? (Phase 5.)
5. Which upstream commit of each repository is compatible with a modern Python 3.12 / arm64 toolchain? (Phase 2–3.)

## 10. Standing prohibitions

- No `sudo`
- No system configuration changes
- No modification of unrelated pre-existing environments, including `/Users/atik/atik/venv/`
- No modification of anything under `research/upstream/`
- No global package installation — isolated environments only
- No credentials in git

# Environment Plan

STATUS: **PHASE 3 COMPLETE — ONE VALIDATED ENVIRONMENT OF RECORD EXISTS (`pixelforge-sam2-v2`).**

The first isolated environment has been created and validated: **`pixelforge-sam2-v2`** (Python 3.11.15, torch 2.13.0), in which SAM 2.1 Hiera-Tiny executed real segmentation inference on Apple MPS. A pre-existing environment named `pixelforge-sam2` — no `-v2` — was found on the host, audited, and **disqualified**: it is bound to the deleted project tree. Details in §11 and in [`docs/experiments/SAM2_MPS_VALIDATION.md`](experiments/SAM2_MPS_VALIDATION.md).

Sections 1–10 describe the rules that govern environment creation. Section 11 records what was actually measured.

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
| PyTorch | **Installed by this project at 2.13.0** in `pixelforge-sam2-v2` (§11). Also present in the disqualified pre-existing environment. Verified absent from `/Users/atik/atik/venv/bin/python`. |
| MPS availability | **MEASURED AND CONFIRMED WORKING, but per-process.** In `pixelforge-sam2-v2` from an interactive session: `is_built()` = `True`, `is_available()` = `True`, Metal device creatable, and a real tensor operation returned the arithmetically expected value. In an earlier non-interactive session on the same host, `is_available()` = `False` because `MTLCreateSystemDefaultDevice()` returned NULL. See §11. |
| Pre-existing interpreter | Python 3.12.0 at `/Users/atik/atik/venv/bin/python` — belongs to an unrelated pre-existing environment. **Not to be used or modified by PixelForge.** |
| PixelForge environments | **One validated:** `pixelforge-sam2-v2`. One pre-existing candidate found and disqualified (§11). |

Metal/MPS is listed in §1 as observed *hardware capability*. That the hardware supports Metal is not the same claim as PyTorch MPS working for a given operator set — and, as §11 shows, it is not even the same claim as Metal being reachable from a given process. All three are distinct and must be recorded separately.

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

| Phase | Environment action | Weights | Outcome |
|---|---|---|---|
| 1 | **None** — plan only | **None** | Done |
| 2 | None. Clone and pin repositories only. | None | Done |
| 3 | Create isolated SAM 2 environment; verify MPS | SAM 2.1 Hiera-Tiny only | **DONE** — `pixelforge-sam2-v2` created, checkpoint acquired (148.78 MiB), MPS inference validated `PASS` (§11) |
| 4 | Create isolated Moebius environment | Minimal Moebius inference checkpoint | In progress (Phase 4) |
| 5 | Create PixelHacker environment, local or cloud per audit | Minimal PixelHacker inference checkpoint | Not started |
| 6+ | Reuse validated environments; add only as needed | As validated | Not started |

## 9. Known open questions

Unresolved, to be answered by measurement rather than assumption:

1. Does PyTorch MPS work for the operator sets these models require? (Phase 3 onward, per model.) — **ANSWERED FOR SAM 2: yes.** The full SAM 2.1 Hiera-Tiny image-inference operator set executed on MPS with no fallback and no error, producing correct masks. This is a per-model answer and does not generalize to the other six.
2. What is the maximum inpainting resolution that fits in 18 GB, per backend? (Phase 4 onward.)
3. Can a segmentation model and an inpainting model be co-resident, or must the pipeline serialize `load()`/`unload()`? (Phase 7.) — **PARTIAL INPUT MEASURED:** SAM 2.1 Hiera-Tiny holds 1205.92 MiB driver-allocated against a 12288.02 MiB recommended ceiling, so ~11 GiB remains for an inpainting model. Whether that is enough depends on the backend, which Phase 4 measures.
4. Does PixelHacker have a viable non-CUDA inference path? (Phase 5.)
5. Which upstream commit of each repository is compatible with a modern Python 3.12 / arm64 toolchain? (Phase 2–3.) — **ANSWERED FOR SAM 2:** the pinned commit runs correctly under Python **3.11.15** with torch 2.13.0 / torchvision 0.28.0 / numpy 2.4.6 on arm64, satisfying all 7 lower-bound-only requirements.
6. Is Metal reachable from the execution contexts this project actually runs in? (New, raised by Phase 3.) — **ANSWERED: it depends on the context, and both outcomes were observed on this one host.** An interactive session reached Metal and ran the inference; a non-interactive session got `MTLCreateSystemDefaultDevice()` → NULL. Hardware capability, PyTorch MPS support, and per-process Metal reachability are three separable facts, and every device audit must record all three.

## 10. Standing prohibitions

- No `sudo`
- No system configuration changes
- No modification of unrelated pre-existing environments, including `/Users/atik/atik/venv/`
- No modification of anything under `research/upstream/`
- No global package installation — isolated environments only
- No credentials in git

## 11. Measured facts (Phase 3, 2026-08-24)

Everything in this section was measured. Nothing is estimated. The full record is [`docs/experiments/SAM2_MPS_VALIDATION.md`](experiments/SAM2_MPS_VALIDATION.md); the machine-readable form is `outputs/sam2_mps_validation/result.json` (git-ignored).

### The validated environment of record

| Property | Value |
|---|---|
| Name | **`pixelforge-sam2-v2`** |
| Path | `/opt/anaconda3/envs/pixelforge-sam2-v2` |
| Python | **3.11.15** |
| torch / torchvision / numpy | **2.13.0 / 0.28.0 / 2.4.6** |
| Editable installs | **None** — verified, per the acceptance rule below |
| Result under it | SAM 2.1 Hiera-Tiny MPS inference `PASS` |

### Device audit — three separable facts, both outcomes observed

| Fact | Interactive session (validating) | Earlier non-interactive session |
|---|---|---|
| Metal GPU present in hardware | **Yes** — Apple M3 Pro | Yes |
| `torch.backends.mps.is_built()` | **`True`** | `True` |
| `torch.backends.mps.is_available()` | **`True`** | **`False`** |
| `MTLCreateSystemDefaultDevice()` | **creatable** | **NULL** |
| `torch.ones((2,2), device="mps")` | **succeeded, value matched expected** | `RuntimeError: The MPS backend is supported on macOS 14.0+` |
| `sw_vers -productVersion` | 15.7.9 (build 24G830) | 15.7.9 |
| `torch._C._mps_is_on_macos_or_newer(15, 0)` | `True` | `True` |

**The PyTorch error message from the failing session was misleading and must not be quoted as a diagnosis.** It named macOS 14.0 while the host runs 15.7.9 and PyTorch's own OS probe agreed. The real cause was that the process had **no Metal device**, which `MTLCreateSystemDefaultDevice()` establishes directly. The same code, same host, same environment then passed from an interactive session.

Operational consequences, both retained as standing rules:

1. **A device audit must probe Metal directly, not trust `is_available()`'s error text.** Record hardware capability, PyTorch MPS support, and per-process Metal reachability as three separate fields.
2. **Metal reachability is a property of the process, not the machine.** Validation must be run from a context that can actually reach the GPU. Background, sandboxed, and non-interactive contexts may not be able to, through no fault of the code.

### Measured performance and memory

| Metric | Value |
|---|---|
| Model load | **0.2975 s** (38.96 M params, `mps:0`, float32) |
| Warm inference total | **0.1508 s** (`set_image` 0.1421 s + `predict` 0.0086 s) |
| Cold inference total | 0.5710 s |
| Process peak RSS | **789.41 MiB** |
| MPS current allocated | 313.64 MiB |
| MPS driver allocated | **1205.92 MiB** |
| MPS recommended max | **12288.02 MiB** |
| Unified memory | 19327352832 B (18 GiB) |

Two facts that shape the architecture:

- **Warm `predict` is 23× faster than cold** (0.0086 s vs 0.2018 s) — Metal shader compilation is a one-off. Latency budgets must distinguish first-call from steady-state.
- **Warm latency is essentially resolution-independent** (0.1508 s at 640×480 vs 0.1554 s at 1800×1200), because SAM 2 resizes to a fixed 1024×1024 internal resolution. The encoder dominates; an extra click costs ~0.009 s. **Encode once per image, cache the embedding, serve clicks from it.**
- **The recommended MPS ceiling is 12288.02 MiB, not 18 GiB.** That is the number to design co-residency against, and SAM 2 uses ~10 % of it.

### Network reachability — a per-session property

From the non-interactive session, every package index and the checkpoint host were unreachable (`pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`, `dl.fbaipublicfiles.com` all HTTP `000`; `pip download` → `Tunnel connection failed: 403 Forbidden`); only `agentrouter.org` and `api.anthropic.com` were allowlisted. Environment creation and checkpoint acquisition were therefore performed from an interactive terminal instead.

**Standing consequence: package installation and weight downloads are handoff items** for any session that cannot reach an index. This is an execution-context limitation, not a project blocker.

### Disqualified pre-existing environment

A conda environment named `pixelforge-sam2` (no `-v2`) exists at `/opt/anaconda3/envs/pixelforge-sam2`, created **2026-08-24 04:02:18 +0600** via `conda create -n pixelforge-sam2 python=3.11 -y` — hours before the Phase 1 clean rebuild.

It contains an **editable install bound to the deleted project tree**:

```
direct_url.json -> {"dir_info": {"editable": true},
                    "url": "file:///Users/atik/atik/venv/Pixelhacker/research/upstream/sam2"}
```

`pip` lists `SAM-2 1.0` as installed; its target no longer contains the package, so bare `import sam2` raises `ModuleNotFoundError`. **This is the previous, deleted environment. It was not adopted; `pixelforge-sam2-v2` was created instead.**

Shadowing risk was checked and excluded: `sys.meta_path` order is `DistutilsMetaFinder → BuiltinImporter → FrozenImporter → PathFinder → _EditableFinder`, so `PathFinder` resolves first and a `sys.path` insertion of the pinned checkout wins.

**Rule added by this finding:** before an environment is accepted as the environment of record, verify it contains **no editable install and no `*.pth` path injection** pointing outside the current project. Record `pip list` and the contents of any `__editable__*` finder.

### Reading upstream code without installing it

Established during Phase 3 and adopted as the standing pattern for READ-ONLY upstream trees:

```python
sys.dont_write_bytecode = True          # before any upstream import
sys.path.insert(0, "<pinned checkout>") # no pip install
```

Verified consequence: `import sam2` succeeds, and the upstream checkout stays byte-identical — 0 changed files before and after, and 0 `__pycache__` / `*.egg-info` directories created. This held through the full inference run.

This is preferred over `pip install -e .` for three reasons, all specific to SAM 2's packaging: an editable install writes `*.egg-info` into the read-only tree; `pyproject.toml` declares `torch>=2.5.1` as a *build* requirement, so default PEP 517 build isolation downloads a second torch; and `setup.py` builds a CUDA-only `sam2._C` extension whose failure is silently swallowed (`SAM2_BUILD_ALLOW_ERRORS` defaults to `"1"`), so a "successful" install can hide a skipped extension.

### Dependency facts read from the pinned SAM 2 commit

`python_requires>=3.10.0`, no ceiling. Exactly 7 runtime requirements, all lower-bound-only: `torch>=2.5.1`, `torchvision>=0.20.1`, `numpy>=1.24.4`, `tqdm>=4.66.1`, `hydra-core>=1.3.2`, `iopath>=0.1.10`, `pillow>=9.4.0`. The three extras (`notebooks`, `interactive-demo`, `dev`) are not needed for image inference — this satisfied the "do not blindly install a giant dependency set" constraint with a 7-package install.

`sam2._C` is **not required for image inference** and was confirmed absent at runtime (`c_extension_available` = `False`) with no effect on the result: its single import site is function-local inside `get_connected_components`, reached only when `max_hole_area > 0` or `max_sprinkle_area > 0` (both default `0.0` for `SAM2ImagePredictor`) and additionally wrapped in `try/except`.

### Dtype policy — confirmed by measurement

§1 notes that `float16`/`bfloat16` support on MPS is uneven. For SAM 2 image inference the question does not arise: neither `set_image` nor `predict` uses `torch.autocast`, and no dtype coercion occurs on the image path. Measured parameter dtypes were `['torch.float32']` exactly. **float32 is the upstream default, not a choice imposed by PixelForge.** Every autocast example in the repository is hardcoded to `device_type="cuda"`; the `bfloat16` casts are confined to the video predictors. Running float32 on MPS is faithful to the published method under §7's "legitimate accommodations" rule.

### Checkpoint identity

| Property | Value |
|---|---|
| File | `checkpoints/sam2/sam2.1_hiera_tiny.pt` (git-ignored) |
| Source | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` |
| Size | 156008466 B (148.78 MiB) |
| Observed SHA-256 | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| Upstream-published checksum | **none exists** |

The SHA-256 is a **local re-acquisition anchor**, not verification against an upstream value. Weights were verified genuinely loaded by comparing 470 checkpoint tensors element-wise against the built model's parameters (0 mismatched) — necessary because `build_sam2` accepts `**kwargs` it never reads, so a dropped `ckpt_path` would otherwise yield a silently random-initialized model.

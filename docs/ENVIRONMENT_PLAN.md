# Environment Plan

STATUS: **PHASE 5 COMPLETE AS A SOURCE CLASSIFICATION — NO PIXELHACKER ENVIRONMENT CREATED. TWO VALIDATED ENVIRONMENTS OF RECORD REMAIN: `pixelforge-sam2-v2` (SAM 2, `PASS`) AND `pixelforge-moebius` (MOEBIUS, `CONDITIONAL`). PIXELHACKER IS `CLOUD_GPU` / `LOCAL_MPS: FAIL` / NOT RUNTIME-VALIDATED.**

The first isolated environment has been created and validated: **`pixelforge-sam2-v2`** (Python 3.11.15, torch 2.13.0), in which SAM 2.1 Hiera-Tiny executed real segmentation inference on Apple MPS. A pre-existing environment named `pixelforge-sam2` — no `-v2` — was found on the host, audited, and **disqualified**: it is bound to the deleted project tree. Details in §11 and in [`docs/experiments/SAM2_MPS_VALIDATION.md`](experiments/SAM2_MPS_VALIDATION.md).

A second environment, **`pixelforge-moebius`**, has been created for Phase 4, passes the same editable-install audit, and has now **executed real generative inpainting on MPS** — verdict **`CONDITIONAL`**, all 11 gate checks passed. It is `CONDITIONAL` rather than `PASS` because student inference on Apple Silicon requires a PixelForge-side import-isolation workaround that does **not** alter the research method. Details in §12 and in [`docs/experiments/MOEBIUS_MPS_VALIDATION.md`](experiments/MOEBIUS_MPS_VALIDATION.md).

Phase 5 classified PixelHacker from the pinned source. **No third environment was created, no PixelHacker UNet weights were downloaded, and no cloud GPU was deployed.** Placement is `CLOUD_GPU`; `LOCAL_MPS` is `FAIL`. Full record: [`docs/experiments/PIXELHACKER_FEASIBILITY.md`](experiments/PIXELHACKER_FEASIBILITY.md) and §13.

Sections 1–10 describe the rules that govern environment creation. Sections 11–12 record what was actually measured, per model. Section 13 records a **source classification**, not a runtime measurement.

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

Cloud execution is deferred until a model actually requires it. **Phase 5 established that PixelHacker requires it** (`LOCAL_MPS: FAIL`; primary classification `CLOUD_GPU`). The worker itself is **not deployed**. When it arrives:

- Provider and instance type recorded alongside the same version facts as local environments
- Credentials via environment variables only — never committed
- The same adapter contract, so the pipeline cannot tell the difference
- No latency or VRAM figure from a cloud run is to be invented in the meantime — see §13 / the feasibility doc for estimated vs unvalidated items

## 8. Phase-by-phase environment schedule

| Phase | Environment action | Weights | Outcome |
|---|---|---|---|
| 1 | **None** — plan only | **None** | Done |
| 2 | None. Clone and pin repositories only. | None | Done |
| 3 | Create isolated SAM 2 environment; verify MPS | SAM 2.1 Hiera-Tiny only | **DONE** — `pixelforge-sam2-v2` created, checkpoint acquired (148.78 MiB), MPS inference validated `PASS` (§11) |
| 4 | Create isolated Moebius environment | Moebius `ft_places2` student + SD VAE only | **DONE** — `pixelforge-moebius` created and audited; both checkpoints downloaded; real MPS inference executed; verdict **`CONDITIONAL`**, 11/11 checks passed (§12) |
| 5 | Classify PixelHacker placement from pinned source; **do not** create an env or download UNet weights until a runtime gate is scheduled | None this phase (VAE already present from Phase 4; PixelHacker UNet **not** downloaded) | **DONE as classification** — `CLOUD_GPU`, `LOCAL_MPS: FAIL`, not runtime-validated (§13) |
| 6+ | Reuse validated environments; add only as needed | As validated | Not started |

## 9. Known open questions

Unresolved, to be answered by measurement rather than assumption:

1. Does PyTorch MPS work for the operator sets these models require? (Phase 3 onward, per model.) — **ANSWERED FOR SAM 2: yes.** The full SAM 2.1 Hiera-Tiny image-inference operator set executed on MPS with no fallback and no error, producing correct masks. This is a per-model answer and does not generalize to the other six.
2. What is the maximum inpainting resolution that fits in 18 GB, per backend? (Phase 4 onward.)
3. Can a segmentation model and an inpainting model be co-resident, or must the pipeline serialize `load()`/`unload()`? (Phase 7.) — **BOTH INPUTS NOW MEASURED, ANSWER LIKELY YES:** SAM 2.1 Hiera-Tiny holds 1205.92 MiB driver-allocated and Moebius holds 4498.22 MiB, against a 12288.02 MiB recommended ceiling. Their sum is 5704.14 MiB — **46.4 %** of the ceiling — so co-residency at 512×512 appears feasible with ~6.4 GiB spare. **Not yet demonstrated:** the two have never been resident in one process simultaneously, and Apple's unified pool means the figures are not guaranteed to add linearly. Phase 7 must measure it directly.
4. Does PixelHacker have a viable non-CUDA inference path? (Phase 5.) — **ANSWERED: no viable MPS path without changing the architecture.** GLA is injected into PixelHacker UNet self-attention and imports `fla` at module load; `fla`/Triton are not installable on this host (Phase 4 measurement). Official entry point is `cuda:0` else `cpu`, with no MPS branch. CPU is a device-string fallback, not a practical PixelForge target, and was not benchmarked. Primary classification **`CLOUD_GPU`**, runtime **NOT VALIDATED**. Details: [`docs/experiments/PIXELHACKER_FEASIBILITY.md`](experiments/PIXELHACKER_FEASIBILITY.md).
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

---

## 12. Measured facts (Phase 4, 2026-08-24) — Moebius

Phase 4 is **complete**, verdict **`CONDITIONAL`**, classification **`LOCAL_MPS`**. Real generative inpainting executed on the M3 Pro GPU and all 11 gate checks passed. Every figure below is copied from the authoritative machine-readable record `outputs/moebius_mps_validation/result.json`. Full record: [`docs/experiments/MOEBIUS_MPS_VALIDATION.md`](experiments/MOEBIUS_MPS_VALIDATION.md).

### The Moebius environment

| Property | Value |
|---|---|
| Name | **`pixelforge-moebius`** |
| Python | 3.11.15 |
| Prefix | `/opt/anaconda3/envs/pixelforge-moebius` |
| Interpreter (ground truth) | `/opt/anaconda3/envs/pixelforge-moebius/bin/python` |
| Editable-install audit | **clean** — no `__editable__*` finder, no outward `direct_url.json` |
| Validation state | **`CONDITIONAL`** — 11/11 checks passed on real MPS |

`pixelforge-sam2-v2` was not modified. The two environments share no packages, which is exactly the isolation §3 requires: Moebius pulls `transformers`, `timm`, `einops`, `opencv-python` and `diffusers`, none of which SAM 2 needs.

**Isolation is now proven, not just intended.** `pixelforge-sam2-v2` raises `ModuleNotFoundError` for `diffusers`, `transformers` and `einops` — the packages Moebius requires. Neither environment can satisfy the other's dependencies, which is the strongest available evidence that they are genuinely separate.

### Environment-label discrepancy — identified and corrected

`result.json` records `python_executable = /opt/anaconda3/envs/pixelforge-moebius/bin/python` but `conda_env = pixelforge-sam2-v2` and `conda_prefix = /opt/anaconda3/envs/pixelforge-sam2-v2`.

**Cause:** the interpreter was invoked by absolute path from a shell where `pixelforge-sam2-v2` was still activated. `CONDA_DEFAULT_ENV` and `CONDA_PREFIX` are *shell* variables set by `conda activate`; running another environment's interpreter directly does not update them.

**Resolution:** the interpreter is authoritative, the conda label is not. Proven by execution — the run imported `diffusers` 0.40.0 and `transformers` 5.15.1 and built a 226 M-parameter diffusers UNet, all of which are impossible in `pixelforge-sam2-v2`. The environment of record for Phase 4 is **`pixelforge-moebius`**. **No measurement is affected and no re-run was required**; only the label was wrong. Phase 3's record is likewise unaffected — `pixelforge-sam2-v2` was activated but never written to.

**Operational rule added for later phases: record `sys.executable` and `sys.prefix` as provenance, never `CONDA_DEFAULT_ENV` alone.** The harness captured all three, which is the only reason the discrepancy was detectable.

| Package | Installed | Upstream pin | Note |
|---|---|---|---|
| torch | 2.13.0 | `2.7.1+cu130` | CUDA wheel does not exist for arm64 |
| torchvision | 0.28.0 | unpinned | |
| diffusers | 0.40.0 | `0.38.0` | deviation, verified by execution |
| transformers | 5.15.1 | unpinned | import-time only; no tokenizer runs |
| accelerate | 1.14.0 | `1.14.0` | matches |
| timm | 1.0.28 | unpinned | |
| einops | 0.8.2 | **absent** | genuine gap in upstream requirements |
| opencv-python | 5.0.0.93 | unpinned | |
| omegaconf | 2.3.1 | unpinned | on the real inference path |
| `flash-linear-attention` | **absent** | `[cuda]==0.3.2` | impossible on Apple Silicon |
| `triton` | **absent** | transitive | no macOS/ARM64 wheels |

### The dependency finding that defines Phase 4

`flash-linear-attention` requires Triton, which publishes no Apple Silicon wheels. It is therefore **not installable on this host at any version**. It is also teacher-only — student inference needs nothing from it — yet it blocks the student import, because `model_lib/__init__.py:6` imports the PixelHacker teacher and Python executes a package's `__init__.py` before any submodule.

Resolved **without modifying upstream**, by seeding `sys.modules['model_lib']` with a surrogate package carrying the real `__path__` and an empty body, then re-exporting exactly what `__init__.py` lines 2–3 export. Verified by execution: `fla`, `triton`, `unet_gla` and `layers.gla` are all absent from `sys.modules` — checked twice, including after `removal.v1_2`'s own `__init__` runs; 13 student symbols exported; upstream 0 changed files, 0 `__pycache__`, before and after.

**This workaround is load-bearing and must be preserved into the adapter layer at Phase 6.** Any code path that imports `model_lib` before the surrogate is installed will pull the teacher and fail on `fla`. It is the sole reason the verdict is `CONDITIONAL` rather than `PASS`, and it cannot be discharged — it is a permanent property of this commit on this platform.

This is a general lesson for the remaining components, not a Moebius quirk: **a research repository's `__init__.py` may couple components that are logically independent.** The dependency audit must trace the actual import graph, not the requirements file. Phase 5 confirmed the **inverse** of the Moebius case: in PixelHacker, GLA is not a skippable teacher import — it **is** the UNet. The same `fla` package that Moebius isolates away is on PixelHacker’s real inference path, which is why isolation cannot be reused and why placement is `CLOUD_GPU`. Grounded-SAM (Phase 11) still ships CUDA-compiled extensions and remains a separate audit.

### Device audit — the per-process condition, now observed from both sides

| Fact | Phase 4 automated session | Phase 4 interactive run |
|---|---|---|
| `torch.backends.mps.is_built()` | True | **True** |
| `torch.backends.mps.is_available()` | **False** | **True** |
| `MTLCreateSystemDefaultDevice()` | **NULL** | **device created** |
| `torch.ones((2,2), device="mps")` | not executed | **executed, result verified correct** |

§11 records that Metal reachability is a property of the **process**, not the machine. Phase 4 confirms it from both sides on the same host, minutes apart: the automated session saw no Metal device, the interactive terminal saw a working one. **The GPU was never the problem.** Every future model gate should assume the automated session may be Metal-less and plan the interactive handoff up front.

Hardware confirmed at run time: `macOS-15.7.9-arm64` (build 24G830), Apple M3 Pro, 11/11 cores, 19 327 352 832 B unified memory.

### Performance and memory — measured

512×512, 20 steps, batch 1, float32, `mps`, `ft_places2` weights. No figure is estimated and no CUDA benchmark informs any of them.

| Metric | Value |
|---|---|
| Student model load (build 1.1892 + weights 0.9443) | **2.1335 s** |
| VAE load | **0.2174 s** |
| First (cold) inference | **21.8105 s** |
| Warm inference (best of 3) | **21.9121 s** |
| Warm inference (mean of 3) | 22.0682 s |
| Per denoising step (warm) | **1.0956 s** |
| Cold / warm ratio | **1.0×** |
| Process peak RSS | **1589.23 MiB** |
| MPS current allocated | 1182.09 MiB |
| **MPS driver allocated** | **4498.22 MiB** |
| MPS recommended max | 12288.02 MiB |
| Headroom used | **36.6 %** of the ceiling |

The design ceiling is `torch.mps.recommended_max_memory()` = **12288.02 MiB** (measured in §11), not the nominal 18 GiB, and RSS and MPS allocator figures **overlap and must not be summed**. Moebius's student is 226 041 531 parameters — **5.8× SAM 2's 38.96 M** — plus an 83 653 863-parameter SD VAE, both float32. Against that ceiling it lands at 36.6 %: **no memory pressure at this resolution**, and failure class F did not occur.

Two results worth carrying forward:

- **There is no cold-start penalty to exploit.** The cold/warm ratio is 1.0×, against SAM 2's 23×. SAM 2's cold cost was one-off shader compilation amortized over an 0.0086 s warm inference; Moebius spends essentially all of its time in 20 sequential UNet steps, which dwarf compilation. **Design consequence: caching or pre-warming Moebius buys nothing.** Latency scales linearly with `num_steps` at ~1.10 s/step, so step count is the only obvious lever — and its quality cost is unmeasured.
- **~21.9 s per image is a hard UX constraint, not a validated target.** It is recorded as a measurement. Nothing here endorses it as acceptable for interactive use, and no optimization (batching, step reduction, float16, quantization) has been evaluated.

### Checkpoint policy for Moebius

Exactly two files, both git-ignored under `checkpoints/moebius/`, **both now downloaded and verified**:

| Role | Size | Source |
|---|---|---|
| Student `ft_places2` | 905 298 356 B (863.36 MiB), sha256 `6525afb8…6a09a` | `https://huggingface.co/hustvl/Moebius/resolve/main/ft_places2/diffusion_pytorch_model.bin` |
| SD VAE (`sdvae_f8d4`) | 167 395 094 B (159.64 MiB) | `https://huggingface.co/hustvl/PixelHacker/tree/main/vae` |

The student checkpoint loaded with `weights_only=True`, yielded 1203 `state_dict` entries, and **1203 of 1203 tensors matched the built model element-wise, 0 mismatched** — so the output is attributable to trained weights, not to a silently-random model. Upstream publishes no checksum for either file, so the recorded SHA-256 is a local re-acquisition anchor, not verification against a published value.

The VAE genuinely comes from the PixelHacker weights repository — upstream's own instruction (`README.md:123-146`), not a substitution. Explicitly **not** downloaded: the base `pretrained` checkpoint, `ft_celebahq`, `ft_ffhq`, the PixelHacker teacher weights, and all training datasets.

Note for §5: the yaml's `vae.model_dir` is `./weight/vae`, **CWD-relative**. It only resolves if the process happens to run from the repo root. PixelForge overrides it to an absolute path outside the READ-ONLY tree — a path fix, not a model change. Expect more CWD-relative paths in research repositories.

### Dtype policy — Moebius

float32, and it is **upstream's own inference default**, not a PixelForge conservatism: `pipeline.py`'s signature default is float16, but `infer/utils.py:79` overrides it to `torch.float`. It is also the correct first attempt on MPS under §1's uneven-float16 note. Measured parameter dtypes were `["torch.float32"]` exactly, for both student and VAE. float16 and bfloat16 on MPS remain untested.

### Network reachability

`huggingface.co`, `pypi.org` and `github.com` all returned HTTP `000` from the automated session, while the interactive terminal downloaded both checkpoints successfully. §11 already records network reachability as a **per-session property**; Phase 4 confirms it independently and from both sides. Checkpoint acquisition, like MPS execution, requires the interactive terminal.

---

## 13. Classified facts (Phase 5, 2026-08-27) — PixelHacker

Phase 5 is **complete as a source classification**, not as a runtime gate. **No environment of record exists for PixelHacker.** Nothing in this section is a latency or VRAM measurement from a PixelHacker process.

Full record: [`docs/experiments/PIXELHACKER_FEASIBILITY.md`](experiments/PIXELHACKER_FEASIBILITY.md).

| Property | Value |
|---|---|
| Commit | `f5567db2871598aa178fe7a34c520dd478a0b41b` (verified twice; working tree clean) |
| Primary classification | **`CLOUD_GPU`** |
| LOCAL_MPS | **FAIL** |
| CLOUD_GPU runtime | **CONDITIONAL / NOT YET RUNTIME-VALIDATED** |
| CPU as PixelForge target | rejected (device-string fallback only; not benchmarked) |
| Environment created | **No** |
| Dependencies installed | **No** |
| UNet weights downloaded | **No** |
| Cloud worker deployed | **No** |
| Adapter implemented | **No** |

### Why LOCAL_MPS fails (source, plus one prior measurement)

PixelHacker’s UNet **is** Gated Linear Attention: `inject_gla_into_tf2dmodel` replaces `tfblock.attn1` in every cross-attention down / mid / up block. `gla_model/gla.py` imports `fla.ops.gla` at module load. That is the research architecture, not an optional teacher.

The Moebius import surrogate **cannot** be reused: skipping GLA here removes PixelHacker, whereas skipping GLA there only skipped a teacher that the student does not run. `load_state_dict(..., strict=False)` is forbidden — it can leave Xavier-initialised GLA modules in the forward pass (`PixelHacker.initialize_weights` runs after injection).

Phase 4 already **measured** that `flash-linear-attention` / Triton are not installable on this M3 Pro. The official PixelHacker entry point is `cuda:0` if CUDA else `cpu`; there is no MPS branch.

`flash-attn==2.5.8` is in `requirements.txt` and is **not imported** by the inference `*.py` files. It is not the MPS blocker.

### Environment action reserved, not executed

When a runtime gate is scheduled, create **`pixelforge-pixelhacker`** as a new isolated environment (Python 3.10 per upstream README, `torch==2.3.0` CUDA wheel, `flash-linear-attention==0.3.2`). Do not merge it with `pixelforge-moebius` (diffusers 0.30.2 vs 0.40.0; torch 2.3.0 vs 2.13.0).

Minimum checkpoint set (HF sizes, **not downloaded**): `ft_places2/diffusion_pytorch_model.bin` (3 449 345 440 B) + `vae/config.json` + `vae/diffusion_pytorch_model.bin` (167 394 306 B). The VAE bytes match the Phase 4 local VAE file; that file does not substitute for the missing UNet.

Reasonable first cloud GPU: **16 GB VRAM minimum, 24 GB preferred** — **ESTIMATED**, not measured. No cloud latency is recorded.

Weight license on Hugging Face is **MIT**; GitHub code license is **Apache-2.0**. Record both; do not assume they are the same as Moebius (Apache-2.0 on code **and** weights).

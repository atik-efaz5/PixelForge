# SAM 2.1 MPS Validation

**Phase 3 — SAM 2.1 MPS reproducibility gate.**

Question this document answers, and the only one it answers: *can the pinned SAM 2 commit run real image segmentation on this machine using MPS?*

Date of validation: **2026-08-24**
Harness: [`tests/smoke/test_sam2_mps.py`](../../tests/smoke/test_sam2_mps.py)
Machine-readable record: `outputs/sam2_mps_validation/result.json` (git-ignored)

> **Answer: yes.** Real segmentation inference executed on the Apple M3 Pro GPU via MPS, loaded from verified trained weights, and produced non-empty masks on two images. Every number below is a measurement taken from `result.json`. Nothing is estimated.
>
> This document was **regenerated from the actual `result.json`** after the harness ran on real hardware, replacing an earlier `PARTIAL` record written when the measuring session had no Metal device. It was not edited to look successful.

---

## Repository

| Property | Value |
|---|---|
| Repository | SAM 2 (`facebookresearch/sam2`) |
| Local path | `/Users/atik/Projects/PixelForge/research/upstream/sam2` |
| Remote | `https://github.com/facebookresearch/sam2.git` |
| Treated as | **READ-ONLY.** Not modified at any point. |
| Working tree | **Clean** — `git status --porcelain` returned 0 lines, verified after every step |
| Installed into the environment? | **No.** See *Import mechanism*. |

### Import mechanism

The pinned checkout is **not** pip-installed. `sys.path.insert(0, <checkout>)` is used instead, with `sys.dont_write_bytecode = True` set before any upstream import.

This is a deliberate correctness decision, not a shortcut:

- `pip install -e .` writes `*.egg-info` into the upstream tree, and importing writes `__pycache__/` directories into it. Both violate the READ-ONLY rule.
- `pyproject.toml` declares `torch>=2.5.1` as a **build-system** requirement, so a default `pip install` triggers PEP 517 build isolation and downloads a second copy of torch into a throwaway build environment. `INSTALL.md` (lines 109–112) documents `--no-build-isolation` as the workaround.
- `setup.py` builds a CUDA-only extension (`sam2._C`, from `sam2/csrc/connected_components.cu`) via `torch.utils.cpp_extension.CUDAExtension`. `SAM2_BUILD_CUDA` defaults to `"1"` and `SAM2_BUILD_ALLOW_ERRORS` defaults to `"1"`, so on a machine with no CUDA the install **silently succeeds with an empty `ext_modules` list** — the warning is invisible unless pip runs with `-v`. Avoiding the install avoids depending on that silent-failure path.

Verified after import and after inference: upstream working tree still 0 changed files, and 0 `__pycache__` / `*.egg-info` directories anywhere in the checkout.

## Commit

| Property | Value |
|---|---|
| Expected pin (`research/upstream/LOCKFILE.md`) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Observed `git rev-parse HEAD` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Second verification pass | `2b90b9f5ceec907a1c18123530e92e794ad901a4` — identical |
| Match | **YES** |
| Branch | `main` |
| Upstream commit date | 2024-12-15T16:47:17-08:00 |
| Upstream commit subject | ``remove `.pin_memory()` in `obj_pos` of `SAM2Base` to resolve and error in MPS (#495)`` |

The pinned commit was **not** changed. Its subject is itself an Apple MPS bug fix; the measurement below now confirms that favourable prior rather than merely noting it.

## Environment

The environment of record is **`pixelforge-sam2-v2`**, created fresh for this phase.

| Property | Value |
|---|---|
| Path | `/opt/anaconda3/envs/pixelforge-sam2-v2` |
| Python | **3.11.15** |
| Python executable | `/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python` |
| Editable installs | **None** — verified absent (see *Acceptance rule* below) |
| Status | **VALIDATED — this is the environment of record for SAM 2** |

### A pre-existing environment was disqualified

An environment named `pixelforge-sam2` (no `-v2`) also exists at `/opt/anaconda3/envs/pixelforge-sam2`, created 2026-08-24 04:02:18 +0600 — roughly seven hours **before** the Phase 1 clean rebuild. It contains an editable install bound to the **deleted** project tree:

```
site-packages/sam_2-1.0.dist-info/direct_url.json
  {"dir_info": {"editable": true},
   "url": "file:///Users/atik/atik/venv/Pixelhacker/research/upstream/sam2"}

site-packages/__editable___sam_2_1_0_finder.py
  MAPPING = {'sam2':     '/Users/atik/atik/venv/Pixelhacker/research/upstream/sam2/sam2',
             'training': '/Users/atik/atik/venv/Pixelhacker/research/upstream/sam2/training'}
```

`pip` reports `SAM-2 1.0` as installed but its target no longer contains the package, so a bare `import sam2` raises `ModuleNotFoundError`. It was **not** adopted, and `-v2` was created instead.

**Acceptance rule added by this finding:** before an environment is accepted as the environment of record, verify it contains **no editable install and no `*.pth` path injection** pointing outside the current project. Record `pip list` and the contents of any `__editable__*` finder.

### Dependency audit

`setup.py` lines 24–32 list exactly 7 runtime requirements, all lower-bound-only with no upper ceilings; `python_requires` is `>=3.10.0`.

| Package | Required | Installed in `pixelforge-sam2-v2` |
|---|---|---|
| torch | `>=2.5.1` | **2.13.0** |
| torchvision | `>=0.20.1` | **0.28.0** |
| numpy | `>=1.24.4` | **2.4.6** |
| tqdm | `>=4.66.1` | 4.70.0 |
| hydra-core | `>=1.3.2` | 1.3.5 |
| iopath | `>=0.1.10` | 0.1.10 |
| pillow | `>=9.4.0` | 12.3.0 |

All 7 satisfy their lower bounds. The three optional extras (`notebooks`, `interactive-demo`, `dev`) are **not** required for image inference and were not installed. A 7-package install satisfies the "do not blindly install a giant dependency set" constraint.

## PyTorch

Measured inside `pixelforge-sam2-v2` **in the session that ran the inference**.

| Probe | Result |
|---|---|
| `torch.__version__` | **`2.13.0`** |
| `torchvision.__version__` | `0.28.0` |
| `numpy.__version__` | `2.4.6` |
| `torch.backends.mps.is_built()` | **`True`** |
| `torch.backends.mps.is_available()` | **`True`** |
| `torch.cuda.is_available()` | `False` |
| `torch._C._mps_is_on_macos_or_newer(13, 0)` | `True` |
| `torch._C._mps_is_on_macos_or_newer(14, 0)` | `True` |
| `torch._C._mps_is_on_macos_or_newer(15, 0)` | `True` |

### Real tensor operation on MPS

| Property | Value |
|---|---|
| Expression | `torch.ones((2,2), device='mps') @ itself + 1.5` |
| Attempted | `True` |
| Succeeded | **`True`** |
| Value | `[[3.5, 3.5], [3.5, 3.5]]` |
| Expected | `[[3.5, 3.5], [3.5, 3.5]]` |
| Matches expected | **`True`** |
| Selected device | **`mps`** |

The value is checked against the arithmetically expected result, not merely for absence of an exception — an operation can complete on MPS and still return wrong numbers.

### A device audit must probe Metal directly

An earlier, non-interactive session on this same host reported `mps.is_available()` = `False` with `RuntimeError: The MPS backend is supported on macOS 14.0+`, while `sw_vers` reported macOS **15.7.9** and PyTorch's own OS probe agreed the host was 15.0-or-newer. The message was generic fallback wording, not a diagnosis. The real cause was that the process had no Metal device at all:

```python
ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal") \
      .MTLCreateSystemDefaultDevice()   # -> NULL in that session
```

In the interactive session that produced this validation, `metal_default_device_creatable` = **`True`**.

**Operational rule retained:** hardware Metal capability, PyTorch MPS support, and per-process Metal reachability are three separable facts and are recorded as three separate fields. Never quote `is_available()`'s error text as a diagnosis.

## Hardware

| Property | Observed value |
|---|---|
| Chip | **Apple M3 Pro** |
| CPU cores | 11 physical / 11 logical |
| Unified memory | **19327352832 bytes** (18 GiB) |
| Platform string | `macOS-15.7.9-arm64-arm-64bit` |
| macOS | 15.7.9 (build 24G830) |
| Machine / processor | `arm64` / Apple M3 Pro |
| CUDA | Not available — no NVIDIA GPU |
| Metal — hardware capability | Present (M3 Pro integrated GPU) |
| Metal — reachable from the measuring process | **Yes** (`metal_default_device_creatable` = `True`) |

## Checkpoint

| Property | Value |
|---|---|
| Variant | SAM 2.1 Hiera-Tiny (smallest — per the smallest-viable-variant rule) |
| Filename | `sam2.1_hiera_tiny.pt` |
| Source URL | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` |
| Source of that URL | `research/upstream/sam2/checkpoints/download_ckpts.sh` lines 40–41 |
| Paired config | `configs/sam2.1/sam2.1_hiera_t.yaml` (`README.md` line 167) |
| Local path | `checkpoints/sam2/sam2.1_hiera_tiny.pt` — confirmed git-ignored |
| **Present on disk** | **YES** |
| Size | **156008466 bytes (148.78 MiB)** |
| Observed SHA-256 | **`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`** |
| Upstream-published checksum | **None exists** |

The URL was read from the pinned repository, not recalled. `download_ckpts.sh` was **not executed**; it would fetch four SAM 2.1 variants, and only Tiny is wanted. No other variant and no video dataset was downloaded.

**The SHA-256 is a local integrity anchor, not verification against an upstream value.** The repository publishes no checksum and no byte size for any checkpoint. The `38.9` figure beside `sam2.1_hiera_tiny` in `README.md` line 167 sits under the column header **`Size (M)`** and is the **parameter count in millions**, not a file size.

### Weights verified genuinely loaded

| Property | Value |
|---|---|
| Parameter count | **38962498 (38.96 M)** |
| Documented count | 38.9 M |
| Matches documented | **`True`** (within ±5%) |
| Checkpoint tensors matched element-wise | **470** |
| Checkpoint tensors mismatched | **0** |
| `weights_verified_loaded` | **`True`** |
| Parameter dtypes | `['torch.float32']` |
| Model device | **`mps:0`** (requested `mps`, matches) |

This check exists because `build_sam2` accepts `**kwargs` it **never reads** — a misspelled or dropped `ckpt_path` is silently discarded and yields a randomly-initialized model with no error. Comparing checkpoint tensors element-wise against the built model's parameters proves the masks below came from trained weights, not random initialization.

`c_extension_available` = **`False`**, and this cannot affect the result: `_C` has exactly one import site, function-local inside `get_connected_components`, reached from `SAM2Transforms.postprocess_masks` only when `max_hole_area > 0` or `max_sprinkle_area > 0` — both default to `0.0` for `SAM2ImagePredictor` — and the call site is additionally wrapped in `try/except`. `INSTALL.md` confirms skipping it loses only hole/sprinkle removal.

## Test Input

Two cases. Only the first is allowed to decide the verdict.

### Case 1 — `synthetic_disc` (**gating**)

Generated locally, never downloaded. The harness writes it, reloads it from disk, and verifies the PNG round-trip is lossless before use.

| Property | Value |
|---|---|
| Path | `outputs/sam2_mps_validation/input_synthetic.png` |
| Resolution | **640 × 480** |
| Dtype | `uint8` (a hard precondition — see below) |
| Content | Solid shaded disc, radius 90 px, centred at (400, 240), on a vertical gradient with a sinusoidal texture, plus one green distractor rectangle at rows 330–430 / columns 60–200 |
| Determinism | **No RNG of any kind.** Closed-form arithmetic over `np.mgrid` only, so the image is byte-identical on every run and every machine. |
| PNG SHA-256 | `69920e8cc7e1585a0f9196063e875183d3c4b79a8010d83f6dd25122f0152ef5` |
| Array SHA-256 | `3711531a192208fd5a9c693088ef8fa2a653f3d52be95352540ce48709a30607` |

`uint8` is enforced rather than assumed: `set_image` runs `torchvision.transforms.ToTensor()`, which applies the `/255` rescale **only** for `uint8` input. A float array would be normalized against the wrong scale and silently degrade the mask, so the harness raises instead.

### Case 2 — `natural_truck` (**non-gating**)

Reads `research/upstream/sam2/notebooks/images/truck.jpg` — a natural photograph already present in the pinned checkout, read-only, nothing downloaded. Resolution **1800 × 1200**.

A solid synthetic disc is weak evidence that real segmentation occurred, since a threshold would also recover it. The photograph is corroboration that the model performs non-trivial segmentation, but it is never allowed to decide the verdict.

**No ground truth exists for either image, so no IoU or other reference-dependent metric is reported.** The mask-area bounds used (1 %–60 %) are a plausibility check that an object-sized region was produced, not an accuracy metric.

## Prompt

| Property | Case 1 | Case 2 |
|---|---|---|
| Prompt type | Single positive point (click) | Single positive point (click) |
| Coordinates `(X, Y)` px | **`(400, 240)`** — disc centre | `(900, 600)` |
| Label | `1` = foreground | `1` = foreground |
| `multimask_output` | `False` → one mask | `False` → one mask |
| Box / mask input | None | None |
| Internal inference resolution | 1024×1024 (config `image_size`; square resize, aspect ratio **not** preserved) | same |

Coordinate convention read from the `predict()` docstring in `sam2/sam2_image_predictor.py`. This is the MVP interaction being validated: `UPLOAD → CLICK → SAM 2 → MASK`.

## Results

**Real MPS inference completed on both cases and produced non-empty masks.**

Gate outcomes, in order:

| # | Stage | Outcome | Evidence |
|---|---|---|---|
| 1 | Repository audit | **PASS** | Pinned commit matched twice; working tree clean; upstream unmodified |
| 2 | Dependency audit | **PASS** | 7 requirements read from `setup.py`; all satisfied in `pixelforge-sam2-v2` |
| 3 | Device audit | **PASS** | `mps.is_built()` = `True`, `mps.is_available()` = `True`, Metal device creatable, real tensor op returned the arithmetically expected value |
| 4 | Checkpoint audit | **PASS** | 148.78 MiB present, SHA-256 recorded, 470 tensors matched element-wise, 0 mismatched |
| 5 | Minimal smoke test | **PASS** | Both cases produced finite, non-empty masks containing their prompt point |
| 6 | Performance test | **PASS** | Cold and warm latency measured with device synchronization |
| 7 | Integration test | **PASS** | Boolean `H × W` mask contract satisfied; PNG artifacts written and re-readable |

Against the directive's eleven PASS criteria:

| # | Criterion | Status |
|---|---|---|
| 1 | Pinned commit verified | **YES** — matched twice |
| 2 | Isolated environment created | **YES** — `pixelforge-sam2-v2`, no editable installs |
| 3 | PyTorch installed | **YES** — 2.13.0 |
| 4 | MPS built = true | **YES** |
| 5 | MPS available = true | **YES** |
| 6 | Real tensor operation succeeds on MPS | **YES** — value matched expected |
| 7 | SAM 2 Hiera-Tiny loads | **YES** — 38.96 M params on `mps:0`, weights verified |
| 8 | Segmentation inference completes on MPS | **YES** — both cases |
| 9 | Non-empty mask produced | **YES** — 25312 px (8.24 %) gating; 635304 px (29.41 %) corroborating |
| 10 | Output artifacts saved | **YES** — 5 files, listed below |
| 11 | Latency measured | **YES** — see `## Performance` |

**Eleven of eleven met.**

### Mask results

| Property | Case 1 `synthetic_disc` (gating) | Case 2 `natural_truck` |
|---|---|---|
| Succeeded | **`True`** | **`True`** |
| Returned masks dtype | `float32` | `float32` |
| Returned masks shape | `[1, 480, 640]` | `[1, 1200, 1800]` |
| Low-res logits shape | `[1, 256, 256]` | `[1, 256, 256]` |
| Predicted IoU score | **`0.978595`** | `0.717186` |
| Mask dtype after cast | **`bool`** | `bool` |
| Mask shape `H × W` | `[480, 640]` | `[1200, 1800]` |
| Mask area | **25312 / 307200 px = 8.2396 %** | 635304 / 2160000 px = 29.4122 % |
| Mask non-empty | **`True`** | `True` |
| Mask bbox `xyxy` | `[311, 151, 489, 329]` | `[84, 280, 1712, 849]` |
| Contains prompt point | **`True`** | `True` |

Case 1's bbox is `178 × 178` px, centred on `(400, 240)` — consistent with the radius-90 disc that was drawn, and the green distractor rectangle at rows 330–430 was **not** included. The mask tracks the prompted object rather than any bright region.

`predict()` returns float32 `0.0`/`1.0` masks in `C×H×W`, not booleans. The harness casts to `bool` explicitly to meet PixelForge's `H × W` boolean mask contract.

### Artifacts

| File | Size |
|---|---|
| `outputs/sam2_mps_validation/input_synthetic.png` | 66782 B |
| `outputs/sam2_mps_validation/synthetic_disc_mask.png` | 1222 B |
| `outputs/sam2_mps_validation/synthetic_disc_overlay.png` | 63731 B |
| `outputs/sam2_mps_validation/natural_truck_mask.png` | 6917 B |
| `outputs/sam2_mps_validation/natural_truck_overlay.png` | 1872829 B |
| `outputs/sam2_mps_validation/result.json` | 9551 B |

All are git-ignored, per the weights-and-outputs-outside-git policy.

### Dtype policy — float32 is upstream's default, not our choice

Neither `set_image` nor `predict` uses `torch.autocast`, and no dtype coercion occurs in the image path. `autocast` appears only in `benchmark.py` (CUDA-hardcoded) and the video predictors; every autocast example in the repository is hardcoded to `device_type="cuda"`. Running float32 on MPS is the **faithful upstream configuration**, not a modification of the research algorithm.

### Guarantees built into the harness

- MPS is required by default. CPU requires an explicit `--allow-cpu-fallback` flag and then **caps the verdict at PARTIAL — never PASS**. That flag was **not** used; `device.selected` = `mps`.
- Weights are verified loaded by element-wise comparison, so a mask can never be attributed to random initialization.
- On any inference error the harness records the exact traceback and extracts the failing operator name, rather than retrying on another device.
- The harness never downloads anything.
- `INSTALL.md` and `README.md` at this commit contain **zero** mentions of Apple Silicon, macOS, MPS, Darwin, or Metal; requirements name Linux only. There was no upstream Apple-Silicon guidance to rely on — this result is the evidence.

## Performance

All figures measured on MPS with device synchronization around each timed region. Warm figures are best-of-3 iterations.

### Model load

| Metric | Value |
|---|---|
| Config | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| Model load time | **0.2975 s** |
| `apply_postprocessing` | `True` |

### Inference latency

| Metric | Case 1 (640×480) | Case 2 (1800×1200) |
|---|---|---|
| `set_image` cold | **0.3692 s** | 0.1591 s |
| `predict` cold | **0.2018 s** | 0.0234 s |
| **Total cold** | **0.5710 s** | 0.1825 s |
| `set_image` warm best | **0.1421 s** | 0.1463 s |
| `set_image` warm mean | 0.1622 s | 0.1468 s |
| `predict` warm best | **0.0086 s** | 0.0091 s |
| `predict` warm mean | 0.0091 s | 0.0093 s |
| **Total warm best** | **0.1508 s** | 0.1554 s |

Cold-vs-warm on case 1 shows the expected one-off Metal shader compilation and allocator warm-up cost: `predict` drops from 0.2018 s to 0.0086 s, a **23×** improvement. Case 2's cold figures are already warm because it ran second in the same process.

The image encoder (`set_image`) dominates warm latency at ~0.14–0.15 s, while an additional click (`predict`) costs ~0.009 s. **Operational consequence for the MVP: encode once per uploaded image, then serve repeated clicks from the cached embedding.** Interactive re-prompting is effectively free.

Warm total latency is nearly identical at 640×480 and 1800×1200 (0.1508 s vs 0.1554 s) because both are resized to the config's fixed 1024×1024 internal resolution. **Input resolution does not drive SAM 2 latency here.**

### Memory

| Metric | Value |
|---|---|
| Process peak RSS | **827752448 B (789.41 MiB)** |
| MPS current allocated | 328872192 B (313.64 MiB) |
| MPS driver allocated | 1264500736 B (1205.92 MiB) |
| MPS recommended max | 12884918272 B (12288.02 MiB) |
| Unified memory total | 19327352832 B (18 GiB) |

**Apple Silicon shares one memory pool between CPU and GPU, so process RSS and MPS allocator figures overlap and must not be summed.** Against the 12288.02 MiB the driver recommends as a ceiling, SAM 2.1 Hiera-Tiny's 1205.92 MiB driver footprint uses roughly **10 %** — leaving substantial headroom for a co-resident inpainting model, which is the Phase 7 co-residency question.

## Limitations

1. **Single host, single environment.** Every figure comes from one Apple M3 Pro (18 GB) under `pixelforge-sam2-v2` with torch 2.13.0. Nothing here generalizes to other Apple silicon tiers, other memory sizes, or other torch versions.
2. **No accuracy claim, and no dataset.** One deterministic synthetic image plus one upstream sample photo. There are **no ground-truth masks**, so no IoU or reference-dependent metric is reported, and this is explicitly **not** real-world validation. The `0.978595` and `0.717186` figures are the model's **own predicted** IoU estimates, not measured agreement with any reference.
3. **Smallest variant only.** Hiera-Tiny alone. Nothing is established about Small, Base+, or Large — in particular their memory behaviour against the 12288.02 MiB ceiling.
4. **Image path only.** `SAM2ImagePredictor` was exercised. The video predictors were not, and they differ materially: they use `bfloat16` autocast, whose MPS operator coverage is untested here.
5. **`_C` extension absent.** Hole and sprinkle removal are unavailable. Harmless at `max_hole_area = max_sprinkle_area = 0.0`, but a future configuration that raises either would hit an untested path.
6. **Warm figures are best-of-3.** A small sample on a machine with other processes running. Treat them as an order-of-magnitude result, not a benchmark.
7. **Latency was not measured under memory pressure.** Peak RSS stayed near 789 MiB with no swapping. Co-resident models could change latency substantially, and that is unmeasured.
8. **`natural_truck` is corroboration, not a gate.** Its 29.41 % mask area and 0.717186 score were never allowed to decide the verdict.
9. **Metal reachability is per-process.** An earlier non-interactive session on this same host could not reach Metal at all. The classification below holds for interactive execution contexts; a background or sandboxed context may still see `mps.is_available()` = `False` through no fault of the code.

## Verdict

### `PASS`

**Real segmentation inference executed on the Apple M3 Pro GPU via MPS, from verified trained weights, and produced valid non-empty masks. Eleven of eleven PASS criteria are met, with no CPU fallback and no modification to the research algorithm.**

`device.selected` = `mps`; `model_device` = `mps:0`; `--allow-cpu-fallback` was not used. The harness caps the verdict at `PARTIAL` whenever CPU is used, so `PASS` is only reachable through genuine MPS execution.

**Classification: `LOCAL_MPS`.**

Phase 3 is **complete**. This is the project's first validated component and the first half of the MVP path:

```
UPLOAD → CLICK → SAM 2 SEGMENTATION → EDITABLE MASK   ← this half is now validated
  → ONE VALIDATED INPAINTING BACKEND → RESULT
```

### What this unblocks

- The MVP's segmentation stage has a validated backend with measured latency (~0.15 s warm) and a measured footprint (~1.2 GiB driver-allocated).
- The **boolean `H × W` mask contract** is confirmed satisfiable from real SAM 2 output, so downstream adapters can rely on it.
- The remaining MVP blocker is one validated inpainting backend — Moebius at Phase 4, else PixelHacker at Phase 5, else BrushNet.
- Roughly 11 GiB of the driver-recommended ceiling remains unused, which is the input to the Phase 7 co-residency question.

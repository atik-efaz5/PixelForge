# Moebius MPS Validation

**Phase 4 — Moebius student inference reproducibility gate on Apple MPS.**

This document answers exactly one question: *can Moebius student inference produce a valid inpainting result on M3 Pro MPS without changing the research algorithm?*

**Answer: yes, subject to one import-isolation workaround. Verdict `CONDITIONAL`, classification `LOCAL_MPS`.**

Real generative inpainting executed on the Apple M3 Pro GPU via MPS, from verified trained weights, and produced a valid non-empty 512×512 RGB result — 21.9121 s warm, 4498.22 MiB driver-allocated, no CPU fallback, and **no modification to the research algorithm**. All 11 mandated checks passed.

The verdict is `CONDITIONAL` rather than `PASS` for one specific, unavoidable reason: the student cannot be imported on Apple Silicon without a PixelForge-side import-isolation strategy that prevents the CUDA-only PixelHacker teacher/GLA path from loading. That workaround is documented in full under `## Import strategy`.

This document is **regenerated from** `outputs/moebius_mps_validation/result.json` (git-ignored), which is the authoritative machine-readable record. Every figure below is a measured value copied from it. Harness: [`tests/smoke/test_moebius_mps.py`](../../tests/smoke/test_moebius_mps.py).

---

## Repository

| Property | Value |
|---|---|
| Path | `research/upstream/Moebius` |
| Branch | `main` |
| Role | Candidate inpainting backend, **Priority A** |
| License | `Apache-2.0` — `README.md:186`, stated to cover code **and** pretrained weights |
| Installed into the environment | **No** (`installed_into_environment: false`) |
| Import mechanism | `sys.path` insertion; `sys.dont_write_bytecode = True` set before any upstream import |
| Working tree **before** run | **clean** — 0 changed files |
| Working tree **after** run | **clean** — 0 changed files |
| `__pycache__` / `*.egg-info` before | **0** |
| `__pycache__` / `*.egg-info` after | **0** |
| `upstream_unmodified` | **`true`** |

The repository was treated as READ-ONLY throughout, and this is verified as a **postcondition**, not merely a precondition. The harness re-runs `git status --porcelain` and re-counts `__pycache__` directories after the import stage, after pipeline construction, and at the end of the run. All three passes returned clean. No upstream file was edited, moved, patched, or monkeypatched.

Deliberately **not** pip-installed. Installing it would place a mutable copy in `site-packages`, and any later edit there would be invisible to `git status` in the clone — the audit trail would break.

## Commit

```
b88d462bacb9af6e7128a3b4cc4a07418bedfd61
```

Read with `git -C research/upstream/Moebius rev-parse HEAD`, **verified twice within the run** (`observed_commit` and `observed_commit_second_pass`), re-read once more after inference (`observed_commit_after_run`), and matched against the Phase 2 pin in `research/upstream/LOCKFILE.md`. `commit_matches_pin: true`.

The commit was not changed at any point during Phase 4. No mutable tag or branch name is treated as authoritative.

## Environment

| Property | Value |
|---|---|
| **Interpreter (ground truth)** | **`/opt/anaconda3/envs/pixelforge-moebius/bin/python`** |
| Environment name | **`pixelforge-moebius`** |
| Python | **3.11.15** |
| Device | `mps` |
| `pixelforge-sam2-v2` | **not modified** |

### Environment-label discrepancy — identified and resolved

`result.json` records a genuine inconsistency that must not be papered over:

```json
"python_executable": "/opt/anaconda3/envs/pixelforge-moebius/bin/python",
"conda_env":         "pixelforge-sam2-v2",
"conda_prefix":      "/opt/anaconda3/envs/pixelforge-sam2-v2"
```

The terminal session therefore *labelled* itself `pixelforge-sam2-v2` while actually executing the `pixelforge-moebius` interpreter.

**Cause.** The handoff command invoked the interpreter by absolute path (`/opt/anaconda3/envs/pixelforge-moebius/bin/python …`) from a shell in which `pixelforge-sam2-v2` was still activated. `CONDA_DEFAULT_ENV` and `CONDA_PREFIX` are *shell* variables set by `conda activate`; invoking another environment's interpreter directly does not update them. So the harness read a stale shell label while running under a different interpreter.

**Which is authoritative.** `sys.executable` and `sys.prefix` are reported by the running interpreter about itself and cannot be stale. `CONDA_DEFAULT_ENV` is an inherited shell string with no binding to the process. **The interpreter is authoritative; the conda label is not.**

**Proven, not asserted.** `pixelforge-sam2-v2` cannot import the packages this run demonstrably used:

| Package | in `pixelforge-moebius` | in `pixelforge-sam2-v2` |
|---|---|---|
| torch | 2.13.0 | 2.13.0 |
| **diffusers** | **0.40.0** | **ABSENT — `ModuleNotFoundError`** |
| **transformers** | **5.15.1** | **ABSENT — `ModuleNotFoundError`** |
| **einops** | **0.8.2** | **ABSENT — `ModuleNotFoundError`** |

The run imported diffusers 0.40.0 and transformers 5.15.1 successfully and built a 226 M-parameter diffusers UNet. That is impossible in `pixelforge-sam2-v2`. The validation ran in **`pixelforge-moebius`**.

**Consequence for this record.** Environment metadata is corrected here to name `pixelforge-moebius`. **No measurement is affected and no re-run is required** — the interpreter, and therefore every package version, device, and timing, was `pixelforge-moebius` all along. Only the label was wrong.

**Consequence for the Phase 3 record.** None. `pixelforge-sam2-v2` was activated but never written to; SAM 2's validation stands unchanged. The two environments remain isolated — the table above is itself the proof, since neither can satisfy the other's dependencies.

**Operational lesson, recorded for later phases.** `CONDA_DEFAULT_ENV` is not a reliable provenance field when interpreters are invoked by absolute path. `sys.executable` and `sys.prefix` are. The harness already captured all three, which is the only reason the discrepancy was detectable at all — a lesson worth preserving: **record the interpreter, not just the label.**

### Measured package versions

Read from the running interpreter, not from a requirements file.

| Package | Installed | Upstream `requirements.txt` |
|---|---|---|
| torch | **2.13.0** | `2.7.1+cu130` |
| torchvision | 0.28.0 | unpinned |
| numpy | 2.4.6 | unpinned |
| **diffusers** | **0.40.0** | `0.38.0` |
| transformers | 5.15.1 | unpinned |
| accelerate | 1.14.0 | `1.14.0` |
| timm | 1.0.28 | unpinned |
| einops | 0.8.2 | **absent from requirements.txt** |
| omegaconf | 2.3.1 | unpinned |
| safetensors | 0.8.0 | unpinned |
| Pillow | 12.3.0 | unpinned |
| opencv-python | 5.0.0 | unpinned |
| `flash-linear-attention` | **absent** (`fla_installed: false`) | `[cuda]==0.3.2` |
| `triton` | **absent** (`triton_installed: false`) | transitive via `fla` |

Three deviations from `requirements.txt`, each deliberate and each now **validated by a successful run**:

1. **`torch==2.7.1+cu130` was not installed.** That is a CUDA wheel; it does not exist for `arm64` macOS. torch 2.13.0 (ARM64/MPS) was installed instead and executed the full denoise loop successfully.
2. **diffusers 0.40.0 instead of the `0.38.0` pin.** The student imports version-fragile internal diffusers paths, so this needed empirical verification rather than assumption. **Now verified end to end**: the model builds, loads 1203 tensors with `<All keys matched successfully>`, and completes 20 denoising steps. The `0.38.0` pin is not required for this inference path on this host.
3. **`einops` installed although absent from `requirements.txt`.** `model_lib/nets/layers/utils.py:3` imports it directly; upstream only receives it transitively through `flash-linear-attention`. Since that package is excluded here, `einops` must be installed explicitly. This is a genuine gap in the upstream requirements file, not a PixelForge addition.

## Dependency audit

Performed **before** installing anything, by tracing the actual import graph of `python -m infer.infer_moebius` and the student architecture. The governing rule: *a package listed in `requirements.txt` is not assumed to be required by the inference path.* The successful run confirms the audit was correct — nothing excluded turned out to be needed.

### Required by student inference — with file:line evidence

| Package | Reached from |
|---|---|
| torch | throughout |
| torchvision | `removal/v1_2/pipeline.py:10` |
| diffusers | `removal_model.py:46` (`UNet2DConditionModel`), `AutoencoderKL`, `DDIMScheduler` |
| accelerate | `infer/utils_infer.py:5` |
| transformers | `removal/v1_2/__init__.py:3` → `dataset.py:12` → `utils.py:17` (Bert/T5/CLIP tokenizers) |
| timm | `model_lib/nets/layers/_efficientnet_blocks.py:16`, `layers/sana/basic_modules.py:24` |
| einops | `model_lib/nets/layers/utils.py:3` |
| opencv-python | `removal/v1_2/pipeline.py:4` |
| Pillow, numpy | image and mask handling throughout |
| pyyaml | `removal_model.py:37` (`load_cfg`, string-path branch) |
| omegaconf | `removal_model.py:41` (`load_cfg`, dict branch) |
| tqdm | `pipeline.py` denoise loop |
| safetensors | diffusers weight loading |

`transformers` is required for a non-obvious reason worth recording: nothing in the *denoising* path needs it. It is pulled in because `removal/v1_2/__init__.py:3` imports `.dataset`, which imports tokenizers at module scope. Since `removal.v1_2` is imported normally — so upstream's own package init runs verbatim — `transformers` is genuinely on the import path even though **no tokenizer was called during inference** (`text_encoder_required: false`).

### Excluded, with the reason — all confirmed unnecessary by the successful run

| Package | Why it is not required |
|---|---|
| **`flash-linear-attention[cuda]==0.3.2`** | Teacher-only, and **impossible on this platform** — see below |
| **`triton`** | Transitive dependency of `fla`; publishes no macOS / Apple-Silicon wheels |
| `tensorboard` | training-only logging |
| `lpips`, `elatentlpips` | training/eval loss functions |
| `pandas`, `scipy` | training data handling |
| `toml`, `orjson` | training config/serialization |
| `matplotlib` | function-local import, reached only when `visualize=True`; inference default is `False` |

### The `flash-linear-attention` finding

This is the central dependency result of Phase 4 and the sole reason the verdict is `CONDITIONAL`.

`fla` is **not merely unnecessary** for student inference — it is **not installable on Apple Silicon at all**, because it requires Triton and Triton ships no macOS/ARM64 wheels.

It nonetheless blocks the student import, proven by execution:

```
removal/v1_2/__init__.py:1  →  removal_model.py:47  from model_lib import *
  →  model_lib/__init__.py:6  from .nets.unet_gla import UNet2DGLAConditionModel
    →  nets/unet_gla.py:16    from .layers.gla.gla import GLSA, GLCA, DEFAULT_GLA_CONFIG
      →  layers/gla/gla.py:16 from fla.ops.gla import chunk_gla, fused_chunk_gla, ...
ModuleNotFoundError: No module named 'fla'
```

`model_lib/__init__.py` is six lines and imports **both halves of the distillation pair** — the Moebius student on lines 2–3 and the PixelHacker *teacher* on line 6. Because Python executes a package's `__init__.py` before any of its submodules, importing the student triggers the teacher's CUDA-only dependency. See `## Import strategy`.

### MPS hazard sweep — static, and borne out by the run

Verified absent from the student path before execution: no `float64` / `.double()` anywhere (the single most common hard MPS failure), no `torch.fft`, no complex tensors, no `cumsum`/`cumprod`, no `grid_sample`, no `scaled_dot_product_attention`, no `torch.linalg`, no `index_put`, no `torch.compile`. All `einsum` calls are two-operand, routed through `_einsum = lambda *a, **k: einsum(*a, **k).contiguous()` at `layers/utils.py:11`.

The identified residual risk — `F.interpolate` on a **`uint8`** tensor at `pipeline.py:288`, default mode `nearest` — **executed successfully on MPS**. It was the most likely single point of failure and it did not fail.

## Checkpoints

Exactly two were downloaded. Both are stored under `checkpoints/moebius/`, which is git-ignored (`.gitignore:49`). **No weight file is committed.**

### 1 — Moebius student, `ft_places2`

| Property | Value |
|---|---|
| File | `checkpoints/moebius/ft_places2/diffusion_pytorch_model.bin` |
| Source | `https://huggingface.co/hustvl/Moebius/resolve/main/ft_places2/diffusion_pytorch_model.bin` |
| Documented at | `research/upstream/Moebius/README.md:159-172` |
| Size | **905 298 356 B (863.36 MiB)** |
| Observed SHA-256 | **`6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a`** |
| Upstream-published checksum | **none exists** |
| `weights_only=True` load | **succeeded** |
| `state_dict` entries | **1203** |
| `state_dict` prefixes | `diff_model`, `embedding_layer` |
| `state_dict` dtypes | `torch.float32`, `torch.int64` |

### 2 — SD VAE (`sdvae_f8d4`)

| Property | Value |
|---|---|
| Directory | `checkpoints/moebius/vae/` |
| Files | `config.json`, `diffusion_pytorch_model.bin` |
| Source | `https://huggingface.co/hustvl/PixelHacker/tree/main/vae` |
| Documented at | `research/upstream/Moebius/README.md:123-146` |
| Size | **167 395 094 B (159.64 MiB)** |
| Upstream-published checksum | **none exists** |

The VAE genuinely comes from the `hustvl/PixelHacker` weights repository — that is upstream's own instruction, not a substitution by PixelForge.

Upstream publishes no checksums for either file, so the recorded SHA-256 is a **local re-acquisition anchor** rather than verification against a published value — the same limitation recorded for SAM 2 in Phase 3.

### Deliberately not downloaded

`downloaded_but_not_required` is empty. Excluded by instruction:

- `hustvl/Moebius` `pretrained` (base, not needed for this gate)
- `hustvl/Moebius` `ft_celebahq`, `ft_ffhq` (other fine-tuned variants)
- PixelHacker **teacher** weights — the teacher is not on the student inference path
- training datasets (Places2, CelebA-HQ, FFHQ)

### Weights verified genuinely loaded

`load_state_dict` returned `<All keys matched successfully>` with **0 missing** and **0 unexpected** keys. Beyond that, the harness compared **1203 checkpoint tensors element-wise** against the built model's parameters: **1203 matched, 0 mismatched**, `weights_verified_loaded: true`.

This check is not redundant. `strict=True` catches a *shape* or *name* mismatch, but a load that silently no-ops would still leave a randomly-initialized model producing plausible-looking noise. The element-wise comparison is what makes the output attributable to **trained weights**.

The checkpoint was also pre-flighted with `torch.load(..., weights_only=True)` **before** any device work, so a checkpoint-format problem would have been classified `E` at the checkpoint stage and could never be misreported later as an MPS operator problem (`C`). It loaded cleanly; the `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` escape hatch was **not needed**.

## Device

Real Metal execution, verified three ways before any model was loaded.

| Fact | Value |
|---|---|
| `torch.backends.mps.is_built()` | **True** |
| `torch.backends.mps.is_available()` | **True** |
| `MTLCreateSystemDefaultDevice()` | **True** (device created) |
| `torch.cuda.is_available()` | **False** |
| `mps_is_on_macos_13_0_or_newer` | True |
| `mps_is_on_macos_14_0_or_newer` | True |
| `mps_is_on_macos_15_0_or_newer` | True |
| Selected device | **`mps`** |

Arithmetic correctness was checked, not just availability:

```
torch.ones((2,2), device="mps") @ itself + 1.5
  → [[3.5, 3.5], [3.5, 3.5]]   matches_expected: true
```

Completing without raising is not the same as being correct, so the harness compares against the expected value and fails the gate on a mismatch.

### Hardware

| Property | Value |
|---|---|
| Platform | `macOS-15.7.9-arm64-arm-64bit` |
| Processor | **Apple M3 Pro** |
| Cores | 11 physical / 11 logical |
| Unified memory | 19 327 352 832 B (18.00 GiB) |
| macOS build | 24G830 |

### No CPU fallback

`no_cpu_fallback: true`. Every model parameter was verified resident on the GPU: student `parameter_devices: ["mps:0"]`, VAE `parameter_devices: ["mps:0"]`, pipeline `input_ids` on `mps:0`. The harness refuses CPU by default; `--allow-cpu-fallback` exists but caps the verdict at `PARTIAL` and can never yield `PASS` or `CONDITIONAL`. It was not used.

**Note on Metal reachability.** Phase 3 recorded that Metal reachability is a property of the **process**, not the machine, and that both outcomes had been observed on this host. Phase 4 confirms it from both sides: the automated session saw `MTLCreateSystemDefaultDevice() → NULL`, while this interactive terminal run saw a working device on the same hardware minutes later. The GPU was never the problem.

## Import strategy

**This section is the reason the verdict is `CONDITIONAL` and not `PASS`.** It is the one compatibility workaround Phase 4 requires, and it is documented in full because the classification depends on it.

### Why it is necessary

`research/upstream/Moebius/model_lib/__init__.py`, verbatim, all six lines:

```python
# Moebius student
from .nets.unet_lambda_prune_lite import UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8
from .nets.unet_lambda_dwconv_blocks import *  # block factories for student

# PixelHacker teacher
from .nets.unet_gla import UNet2DGLAConditionModel
```

Line 6 imports the **teacher**, reaching `fla` → Triton, which cannot be installed on Apple Silicon. Python runs `__init__.py` before any submodule, so `import model_lib.nets.unet_lambda_prune_lite` — the *student* — dies on the *teacher's* CUDA-only dependency. The student itself needs nothing from `fla`.

The directive forbids editing the upstream file. The workaround is therefore entirely on the PixelForge side.

### The mechanism

```python
pkg = types.ModuleType("model_lib")
pkg.__path__ = [str(MOEBIUS_REPO / "model_lib")]   # the REAL path
pkg.__package__ = "model_lib"
sys.modules["model_lib"] = pkg                      # seeded before first import

student = importlib.import_module("model_lib.nets.unet_lambda_prune_lite")
blocks  = importlib.import_module("model_lib.nets.unet_lambda_dwconv_blocks")

setattr(pkg, STUDENT_CLASS, getattr(student, STUDENT_CLASS))   # __init__.py line 2
for n in dir(blocks):                                          # __init__.py line 3
    if not n.startswith("_"):
        setattr(pkg, n, getattr(blocks, n))
```

A surrogate package object carries the **real** `__path__` but an empty body, so `__init__.py` never executes while submodule imports still resolve normally under their true dotted names. Lines 2–3 — the student exports — are then reproduced exactly. Only line 6, the teacher, is omitted.

This works because `model_lib/nets/`, `model_lib/nets/layers/` and everything below are **PEP 420 implicit namespace packages** — they have no `__init__.py` of their own — so nothing further needs intercepting.

Re-exporting is not cosmetic: `removal_model.py:47` does `from model_lib import *`, and `removal_model.py:71` resolves the architecture with `eval(model_type)` against that module's globals. The student class must genuinely be an attribute of `model_lib` or the build fails.

**13 symbols** were exported, exactly matching what lines 2–3 provide:

```
UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8
DWDownBlock2D        DWMidBlock2D         DWMixTFDownBlock2D
DWMixTFMidBlock2D    DWMixTFUpBlock2D     DWTFDownBlock2D
DWTFMidBlock2D       DWTFUpBlock2D        DWUpBlock2D
custom_get_down_block  custom_get_mid_block  custom_get_up_block
```

### Precisely what this workaround does and does not do

| | |
|---|---|
| **Does not** modify upstream Moebius files | `upstream_file_modified: false`; 0 changed files before **and** after |
| **Does not** change the student model architecture | The student source executes verbatim under its true dotted name |
| **Does not** replace or reimplement research operations | `anything_stubbed_or_mocked: false` — no fake module exists, so none can be reachable from the forward pass |
| **Does** only control import resolution | It decides *which modules load*, never *what they compute* |

The omitted symbol is `UNet2DGLAConditionModel` — the teacher. Moebius is a **distilled student**; running student inference does not involve the teacher in any way. Omitting it removes nothing from the inference path.

A stub for `fla` was considered and **rejected**: `layers/gla/gla.py:27` subclasses `fla`'s `GatedLinearAttention`, so a stub would have to fake real numerical behaviour, and it would still execute teacher code. The surrogate avoids both problems by never reaching the teacher at all.

`removal.v1_2` is imported **normally**, so upstream's own package init runs verbatim. Exactly **one** surrogate exists, for exactly one unavoidable reason.

### Verified teacher-free by execution — checked twice

| Check | After surrogate install | After `removal.v1_2` import |
|---|---|---|
| `fla` in `sys.modules` | **False** | **False** |
| `triton` in `sys.modules` | **False** | **False** |
| any `unet_gla` module loaded | **False** | **False** |
| any `layers.gla` module loaded | **False** | **False** |
| verdict | `teacher_free: true` | `teacher_free_after_removal_import: true` |

Contamination is checked **twice** because `removal.v1_2`'s own `__init__` is the import most likely to pull the teacher in indirectly. It did not.

### One further deviation, recorded

`utils_train.build_vae` is **inlined** rather than imported. Importing it pulls `utils_train.py:14` → `library/train_util.py` and `library/chinese_sdxl_train_util.py` — the entire training stack (tensorboard, toml, orjson, CLIP/Bert/T5 tokenizers) — merely to reach a one-line function. Its complete body (`utils_train.py:27-29`) is:

```python
def build_vae(model_cfg: Dict) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_cfg["vae"]['model_dir'])
    return vae
```

That single statement is reproduced verbatim in the harness, and the loaded class is confirmed to be the genuine `diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL`. Everything else — `load_cfg`, `build_removal_model`, `load_removal_model`, `RemovalSDXLPipeline_BatchMode`, `DDIMScheduler` and its arguments — is upstream's own code called with byte-identical literal values.

## Test input

Deterministic, generated by the harness, byte-identical on every run.

| Property | Value |
|---|---|
| Resolution | 512 × 512 |
| Image PNG SHA-256 | `bd50cc9c2dd36f1bcaf3000c467c7b3b068dbca9390235d044131452ec604907` |
| Mask PNG SHA-256 | `74db662acd36c1f34cc60b20fe0fab7d348e3ce741633233d565604f89772387` |
| Image array SHA-256 | `133d02d5e87a1297edf725d8489280086062b278852597fc77c28422c13ec504` |
| Mask array SHA-256 | `c5dcb71e063faaede4dcb57369a3baf3653dee87d6bb93948b25abee1bdae8a5` |
| Determinism | **no RNG** — closed-form arithmetic over `np.mgrid` only |
| Object | shaded disc, r = 74, centre (352, 200) |
| Mask area | **20 081 px = 7.6603 %** |
| Internal mask dtype | `bool`, H × W (PixelForge contract) |
| Delivered mask dtype | `uint8` grayscale `'L'`, values `{0, 255}` |

The scene is a vertical gradient with sinusoidal texture, crossed by two horizontal bars, with a red sphere occluding both. This is deliberately not a flat background: a correct inpaint must **continue visible structure through the removed region**, which a blur could not fake.

### Mask contract — verified from source, then confirmed by the run

Target chain: **SAM 2 boolean mask → `uint8` 0/255 grayscale → Moebius**.

Read from source — `infer/utils_dataset.py:142-183`, `SimpleInferDataset`:

```python
img  = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path).convert("L")
mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
if img.size[0] != self.resolution or img.size[1] != self.resolution:
    img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
```

So: image = PIL **RGB**, square; mask = PIL **`"L"` grayscale**, resized NEAREST. `pipeline.py:188` then binarizes at `255/2`. **The chain is correct as specified**, and the successful run confirms it end to end.

**Mask polarity: WHITE (255) = region to INPAINT; BLACK (0) = region to KEEP.** Established from three independent lines of arithmetic, then proven by execution, and finally confirmed visually in the output:

- `pipeline.py:232` — `mask = np.asarray(input_mask)/255.` → `{0., 1.}`
- `pipeline.py:175-176` — `mask = where(mask>=0.5,1,0)`; `masked_image = image*(1-mask)` → content is **zeroed** where the mask is white, i.e. white is what gets generated
- `pipeline.py:252` — `ours_np = ours_np*m_img + (1-m_img)*img_np` → white takes **generated** pixels, black takes **original** pixels

The `comparison_before_mask_after.png` artifact shows the white disc region replaced and the black region preserved, matching the derivation exactly.

**No ground truth exists for a synthetic scene, so no PSNR, SSIM, LPIPS, FID or any other reference-dependent metric is reported** (`ground_truth_available: false`).

## Inference configuration

Every value is an upstream default, read from `infer/utils.py` argparse defaults, `pipeline.py` signature defaults, or `config/model_cfg/moebius.yaml`. **No value here is a PixelForge tuning choice.**

| Parameter | Value | Source |
|---|---|---|
| batch size | **1** | directive |
| resolution | **512 × 512** | `--resolution`; yaml `data.image_size` |
| steps | **20** | `--num-step` |
| dtype | **float32** | `infer/utils.py:79` passes `dtype=torch.float` |
| guidance scale | 2.5 | `--cfg` |
| noise offset | 0.0357 | `--noise-offset` |
| paste | True | `--pst` |
| compensate | False | `--cps` |
| strength | 0.99 | `pipeline.py.__call__` default |
| `num_embeddings` | 20 | `infer/utils.py:67` |
| scheduler | `DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False)` | `infer/utils.py:71-73` |
| seed | 0 | `pipeline.py:347-350`, `retry=0` |
| device | **`mps`** | the only substantive deviation |

float32 warrants a note: `pipeline.py`'s own signature default is float16, but upstream's inference entry point overrides it to float32. float32 is therefore the *upstream inference default*, not a PixelForge conservatism — and it is also the right first attempt on MPS, where float16 support is less uniform. Measured parameter dtypes were `["torch.float32"]` exactly, for both the student and the VAE.

Two paths differ from an upstream CUDA run, both recorded in `result.json`:

- `device`: `cuda` → `mps` — a plain CLI argument upstream, so this is device placement, not an algorithm change
- VAE directory: `./weight/vae` → `/Users/atik/Projects/PixelForge/checkpoints/moebius/vae` — the yaml path is **CWD-relative** and only resolves if the process runs from the repo root; overriding it to an absolute path outside the READ-ONLY tree is a path fix, not a model change

Determinism: upstream seeds `random`, `np.random` and `torch.manual_seed` with `seed = 0` at the default `retry=0`, so inference is deterministic per-device. **Caveat:** MPS and CPU use different RNG streams, so `torch.manual_seed(0)` does not make MPS output bit-identical to CUDA or CPU output.

### Model as built

| Property | Value |
|---|---|
| Student class | `UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8` |
| Parameters | **226 041 531** (226.04 M) |
| UNet in / out channels | **9 / 4** (4 latent + 1 mask + 4 masked-latent) |
| `sample_size` | 64 |
| `encoder_hid_dim` | 3072 |
| Embedding | `nn.Embedding(20, 3072)` |
| Pipeline class | `RemovalSDXLPipeline_BatchMode` |
| `vae_ds_ratio` | 8 |
| `input_ids` shape / device | `[2, 10]` on `mps:0` (CFG batch-doubling) |

**No text encoder is on the inference path** (`text_encoder_required: false`). `input_ids` are indices into the learned `nn.Embedding(20, 3072)` at `removal_model.py:14-18` — not tokens. No tokenizer, no CLIP, no T5 is ever called during inference, despite `transformers` being an import-time requirement.

The measured 226.04 M is independently corroborated by upstream's own citation block, which titles the paper *"Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance"* — 226.04 M = 0.226 B. The build produced the published architecture, not a misconfigured variant.

VAE as built: `AutoencoderKL`, **83 653 863 parameters (83.65 M)**, `block_out_channels [128, 256, 512, 512]`, `latent_channels 4`, `scaling_factor 0.13025`, downsample ratio 8.

Combined resident model size: **309.70 M parameters** in float32.

## Performance

All figures measured on this host in this run. **Nothing is estimated, and no CUDA benchmark informs any number here.**

### Load

| Stage | Seconds |
|---|---|
| Student architecture build | 1.1892 |
| Student weight load (`load_removal_model`, → MPS, float32) | 0.9443 |
| **Student model load, total** | **2.1335** |
| **VAE load** | **0.2174** |

### Inference — 512×512, 20 steps, batch 1, float32, MPS

| Metric | Seconds |
|---|---|
| **First (cold) inference** | **21.8105** |
| **Warm inference (best of 3)** | **21.9121** |
| Warm inference (mean of 3) | 22.0682 |
| Warm iterations, all | 22.0358, 22.2566, 21.9121 |
| **Per denoising step (warm)** | **1.0956** |
| Cold / warm ratio | **1.0×** |

The cold-to-warm ratio is **1.0×** — the first inference (21.8105 s) is, within noise, indistinguishable from the warm runs, and in this run was marginally faster than the warm best (21.9121 s). This is the opposite of SAM 2's **23×** cold penalty in Phase 3, and the difference is structural, not noise: SAM 2's cold cost was dominated by one-off Metal shader compilation amortized over a very short (0.0086 s) warm inference, whereas Moebius spends essentially all of its time in 20 sequential UNet denoising steps whose arithmetic dwarfs any shader-compilation overhead. Warm-run spread is 0.34 s across three iterations (≈1.6 %), so the measurement is stable.

**Design consequence:** unlike SAM 2, there is no meaningful warm-up benefit to exploit — the first call costs the same as every later call. Moebius inference cost is fixed per invocation and scales linearly with `num_steps` at ~1.10 s/step. Reducing steps is the only obvious latency lever, and its quality cost is **unmeasured**.

### Memory

| Metric | MiB |
|---|---|
| Process peak RSS | **1589.23** |
| MPS current allocated | 1182.09 |
| **MPS driver allocated** | **4498.22** |
| MPS recommended max | 12288.02 |
| Headroom used | **36.6 %** of the recommended ceiling |

Apple Silicon shares one memory pool between CPU and GPU, so **process RSS and MPS allocator figures overlap and must not be summed**. The meaningful ceiling is `torch.mps.recommended_max_memory()` = 12288.02 MiB, not the nominal 18 GiB.

Moebius uses **3.73×** SAM 2's driver allocation (4498.22 vs 1205.92 MiB) and leaves **63.4 %** of the recommended ceiling free at 512×512. Memory exhaustion (failure class F) did not occur and is not close at this resolution. Behaviour at higher resolutions or with batch > 1 is **unmeasured**.

## Output

Real generative inpainting output, verified four ways.

| Property | Value |
|---|---|
| Type | PIL `Image`, mode `RGB` |
| Size | **512 × 512** |
| Array shape / dtype | `[512, 512, 3]`, `uint8` |
| **Finite** | **True** — no NaN, no Inf |
| **Valid dimensions** | **True** |
| Value range | 28 – 230 |
| Mean | 115.9979 |
| Unique values | 203 |
| Is constant | **False** |
| Masked-region mean / std | 114.6918 / 32.8802 |
| Unmasked-region mean | 116.1062 |
| **PNG written** | **True** — 111 261 B |
| **PNG round-trip lossless** | **True** |

The `is_constant: false` check matters: a uniform image would mean inference "succeeded" while producing no content. With 203 distinct values and a masked-region standard deviation of 32.88, the model generated real structure inside the mask.

The masked-region mean (114.69) sits close to the unmasked-region mean (116.11), consistent with the generated content being tonally integrated with its surroundings rather than an obvious patch — and the sphere's distinctive red (which dominated that region in the input) is gone.

Artifacts: `input_synthetic.png`, `input_mask.png`, `output_inpainted.png`, `comparison_before_mask_after.png`, `result.json` — all under `outputs/moebius_mps_validation/` (git-ignored).

**No accuracy claim is made.** `accuracy_claim` in `result.json` reads `"NONE"`. No ground truth exists, so no reference-dependent metric is reported. The `comparison_before_mask_after.png` strip is for human inspection and is **not a metric**. That the result looks coherent is an observation, not validation.

## Failure analysis

**No failure occurred.** `failure_stage`, `failure_type`, `failure_message`, `failure_traceback`, `failing_operation` and `failure_classification` are all `null`. All 11 mandated checks returned `true`.

### The one issue that was found, diagnosed, and resolved

**Class B — import contamination.** `model_lib/__init__.py:6` imports the CUDA-only PixelHacker teacher, blocking the student's import on a platform where `fla`/Triton cannot exist. Diagnosed by execution, resolved by the PixelForge-side import surrogate documented above, **without modifying upstream**, and verified teacher-free twice at runtime.

This is the sole reason the verdict is `CONDITIONAL`. It is a *resolved* issue, not an outstanding one — but it is a permanent property of this commit on this platform, so it cannot be discharged.

### Risks that were identified pre-run and did not materialize

| Risk | Class | Outcome |
|---|---|---|
| `F.interpolate` on a `uint8` tensor, `pipeline.py:288`, mode `nearest` | C | **executed successfully on MPS** |
| Peak memory for 226 M student + 83.65 M VAE, float32 | F | **4498.22 MiB — 36.6 % of the ceiling, no exhaustion** |
| Unsupported MPS operator elsewhere in the student | C | **none encountered** |
| diffusers 0.40.0 vs the `0.38.0` pin, at runtime | A / G | **full denoise loop completed; 1203/1203 tensors matched** |
| dtype coercion / float64 leakage | D | **none — `["torch.float32"]` exactly** |
| Checkpoint format requiring the legacy unpickler | E | **loaded with `weights_only=True`** |

Every pre-identified risk was checked against measurement rather than assumed away. None required patching the research repository, and none forced a CPU fallback.

### Standing rule, unchanged

Had any of these failed, the correct action would have been to **record the failure**, not to patch upstream. If a future workaround were to require changing the research algorithm, the verdict would become `FAIL — requires upstream modification`.

## Verdict

### `CONDITIONAL`

**Classification: `LOCAL_MPS`. Validation: `CONDITIONAL`.**

Moebius student inference runs successfully on Apple MPS and produces a valid inpainting result from verified trained weights, with no CPU fallback and no modification to the research algorithm. **All 11 mandated checks passed.**

| # | Check | Result |
|---|---|---|
| 1 | MPS available | **PASS** — `is_available()` True, Metal device created, arithmetic verified |
| 2 | Student model import works | **PASS** — teacher-free, 13 symbols exported |
| 3 | VAE import works | **PASS** — genuine `diffusers` `AutoencoderKL` |
| 4 | Checkpoint loads | **PASS** — 1203 entries, `weights_only=True` |
| 5 | Model moves to MPS | **PASS** — `parameter_devices: ["mps:0"]` |
| 6 | VAE moves to MPS | **PASS** — `parameter_devices: ["mps:0"]` |
| 7 | Pipeline initializes | **PASS** — `RemovalSDXLPipeline_BatchMode` |
| 8 | Inference executes | **PASS** — 20 steps, no fallback |
| 9 | Output is finite | **PASS** — no NaN/Inf |
| 10 | Output has valid dimensions | **PASS** — 512×512×3 |
| 11 | Output writes as PNG | **PASS** — lossless round-trip |

### Why `CONDITIONAL` and not `PASS`

`PASS` requires that the model run **with no compatibility workaround at all**. Moebius does not meet that bar on Apple Silicon, and cannot: the student's own package `__init__` imports the CUDA-only PixelHacker teacher, whose dependency chain reaches Triton, which has no Apple Silicon build. Student inference is therefore unreachable on this platform without PixelForge-side import isolation.

The directive defines `CONDITIONAL` as *"works only with a clearly documented compatibility workaround that does not alter the research method."* That is exactly this case. The workaround:

- **does not modify upstream Moebius files** — 0 changed files before and after, verified three times per run
- **does not change the student model architecture** — the student source executes verbatim under its true dotted name
- **does not replace research operations** — nothing is stubbed or mocked, so no substitute code is reachable from the forward pass
- **only controls import resolution** — it decides which modules load, never what they compute

**This result must not be described as `PASS`.**

### What this result does and does not establish

**Established:** Moebius student inference executes on Apple M3 Pro MPS at 512×512 / 20 steps / float32 / batch 1, in 21.9121 s warm, using 4498.22 MiB of GPU memory, from trained `ft_places2` weights, producing a valid non-constant 512×512 RGB image, with the research algorithm unmodified.

**Not established:** inpainting quality (no ground truth, no validation dataset, no reference metric), behaviour at other resolutions, step counts, dtypes or batch sizes, behaviour on other Apple Silicon hardware, and any comparison against PixelHacker or BrushNet.

**This is a reproducibility result, not an accuracy result, and not a statement of production readiness.** A 21.9 s single-image latency is a substantial user-experience constraint that the MVP design must account for; it is reported here as a measurement, not endorsed as acceptable.

## Limitations

1. **No accuracy claim whatsoever.** No ground truth, no validation dataset, therefore no PSNR/SSIM/LPIPS/FID or any reference-dependent metric. A visually coherent result is an observation, not validation.
2. **Not production-ready.** 21.9121 s warm per 512×512 image at 20 steps. No batching, streaming, quantization, step reduction or caching has been evaluated. Nothing here endorses the current latency for interactive use.
3. **The import isolation is load-bearing and must be preserved.** Any future code path that imports `model_lib` *before* the surrogate is installed will pull the teacher and fail on `fla`. This constraint must survive into the adapter layer at Phase 6 — it is not an incidental test detail.
4. **One configuration only.** 512×512, 20 steps, float32, batch size 1, `paste=True`, `guidance_scale=2.5`, `strength=0.99`, seed 0. float16 and bfloat16 on MPS are **untested**. Other resolutions and batch sizes are **untested**.
5. **One checkpoint only.** `ft_places2`. The `pretrained`, `ft_celebahq` and `ft_ffhq` variants are out of scope by instruction; nothing here generalises to them.
6. **One synthetic image.** A single deterministic scene with one convex mask covering 7.66 % of the frame. Real photographs, irregular masks, large masks, multiple disjoint regions and edge-adjacent masks are all **untested**.
7. **Best-of-3 warm timing.** Reported warm latency is the fastest of three iterations (mean 22.0682 s, spread ~1.6 %). No load, thermal, or memory-pressure testing was performed; sustained-throughput behaviour is unknown.
8. **Single host, single OS build.** Apple M3 Pro, macOS 15.7.9 (24G830), 18 GB unified memory. Nothing generalises to M1/M2/M4, to other memory configurations, or to other macOS versions.
9. **Metal reachability is per-process.** Both outcomes were observed on this host during Phase 4. A future automated run may see no GPU; that would be a session property, not a regression.
10. **`diffusers` deviates from the upstream pin** (0.40.0 vs `0.38.0`). Verified working end to end here. If later phases observe drift, 0.38.0 is the documented fallback.
11. **The `transformers` requirement is import-time only.** It is on the import path via `removal/v1_2/__init__.py:3`, but no tokenizer is called during inference. Removing it would require bypassing upstream's package init — deliberately not done.
12. **Cross-device determinism does not hold.** Upstream seeds RNG at `retry=0`, but MPS and CPU draw from different streams; bit-identical output across devices should not be expected.
13. **Checksums are local anchors.** Upstream publishes no checksum for either checkpoint, so the recorded SHA-256 values verify re-acquisition on this host, not authenticity against an upstream-published value.
14. **The teacher is entirely untested.** Only the student path was audited. Whether PixelHacker's GLA teacher can run on MPS is a separate question, deferred to Phase 5.
15. **The environment label in `result.json` is wrong and the corrected value is recorded here.** `conda_env` reads `pixelforge-sam2-v2` because of a stale shell variable; the interpreter was `pixelforge-moebius`. No measurement is affected. Future harnesses should treat `sys.executable` as authoritative provenance.
16. **No integration test yet.** This gate is the *minimal smoke test* and *performance test* stages. Moebius is not wired into any pipeline, and the SAM 2 → Moebius handoff has been verified only at the level of the mask contract, not end to end with a real SAM 2 mask.

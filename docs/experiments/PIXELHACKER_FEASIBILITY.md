# PixelHacker Feasibility Classification

**Phase 5 — final local-vs-cloud placement for the pinned PixelHacker implementation.**

This document answers exactly one question: *can the pinned PixelHacker implementation run locally on the Apple M3 Pro using MPS without changing the model architecture?*

**Answer: no. Primary classification `CLOUD_GPU`.**

| Gate | Verdict |
|---|---|
| **LOCAL_MPS** | **FAIL** |
| **CLOUD_GPU** | **CONDITIONAL / NOT YET RUNTIME-VALIDATED** |
| **CPU** | Not a practical PixelForge target |
| **UNAVAILABLE** | Not selected — a CUDA inference path exists in source |

No PixelHacker UNet weights were downloaded. No PixelHacker dependencies were installed. No cloud GPU was provisioned. No inference was executed. Upstream `research/upstream/PixelHacker` was not modified.

Evidence in this document is labelled:

| Label | Meaning |
|---|---|
| **VERIFIED FROM SOURCE** | Read from the pinned checkout or from a named public metadata endpoint |
| **MEASURED** | Observed on this host in a prior validated phase (cited) |
| **ESTIMATED** | Derived, not executed |
| **NOT VALIDATED** | Required for a future runtime gate; not claimed here |

---

## Repository

| Property | Value | Evidence |
|---|---|---|
| Path | `research/upstream/PixelHacker` | VERIFIED FROM SOURCE |
| Remote | `https://github.com/hustvl/PixelHacker.git` | `research/upstream/LOCKFILE.md` |
| **Pinned commit** | **`f5567db2871598aa178fe7a34c520dd478a0b41b`** | `git rev-parse HEAD`, verified twice on 2026-08-27 |
| Branch | `main` (context only; SHA is authoritative) | VERIFIED FROM SOURCE |
| Working tree | **clean** — 0 changed files | `git status --porcelain` empty |
| Installed into an environment | **No** — Phase 5 did not create one | this phase |
| License (code) | **Apache-2.0** — `LICENSE` (verbatim Apache License 2.0) | VERIFIED FROM SOURCE |
| License (weights) | **MIT** — Hugging Face `cardData.license` | VERIFIED FROM SOURCE (HF API, see §H) |

The commit was not changed during Phase 5. No mutable tag is treated as authoritative.

Hugging Face weights-repo revision `012fd343158936a265b8a0ee38a791a7a2841f45` is recorded in `research/upstream/REPOSITORIES.md`. That is an **HF weights revision, not this GitHub source commit**. The two must never be conflated. Reconfirmed via the HF model API on 2026-08-27 (`sha` field).

---

## Verdict in one paragraph

PixelHacker is a `UNet2DConditionModel` whose **self-attention** (`BasicTransformerBlock.attn1`) is replaced, at construction time, with `GatedLinearAttention` from `flash-linear-attention` (`fla`). GLA is the published architecture, not an optional teacher sidecar. `fla` is imported at module load by `gla_model/gla.py` and is not installable on this Apple Silicon host (Triton; no macOS / Metal wheels — **MEASURED** in Phase 4). The official entry point selects `cuda:0` or else `cpu`; it has **no MPS path**. Removing GLA, stubbing `fla`, or loading with `strict=False` would change or randomly initialise the research model. Therefore **LOCAL_MPS fails**. The remaining honest placement is **CLOUD_GPU**, behind a future adapter. That path is **source-feasible and not runtime-validated**.

---

## A. Exact inference dependency chain

**VERIFIED FROM SOURCE.** Traced from `infer_pixelhacker.py` on commit `f5567db`.

```
infer_pixelhacker.py
  device = "cuda:0" if torch.cuda.is_available() else "cpu"     # line 31 — no mps
  load_cfg(config/PixelHacker_sdvae_f8d4.yaml)
  build_model(cfg, 20)                                          # utils.py:38
    PixelHacker(**model_cfg)                                    # gla_model/PixelHacker.py
      UNet2DConditionModel.__init__(...)                        # diffusers UNet
      inject_gla_into_tf2dmodel(...)  × down / mid / up         # PixelHacker.py:134-151
        tfblock.attn1 = GatedLinearAttention(...)               # PixelHacker.py:406-408
          gla_model/gla.py  (module-level)
            from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
            from fla.models.gla.modeling_gla import GatedLinearAttention, ...
            from fla.modules import RMSNorm, ShortConvolution, ...
  model.load_state_dict(torch.load(weight, device))             # default strict=True
  build_vae(cfg) → AutoencoderKL.from_pretrained(vae/)          # utils.py:65-68
  DDIMScheduler(...)                                            # infer_pixelhacker.py:56-58
  PixelHacker_Pipeline(model, vae, scheduler, device, float32)  # infer_pixelhacker.py:60-65
  SimpleInferDataset → pipe(image, mask, 512, 20 steps, ...)
```

Packages on this path, with file:line evidence:

| Package | Reached from |
|---|---|
| torch | throughout |
| torchvision | `infer_pixelhacker.py:21`, `pipeline.py:18` |
| diffusers | `UNet2DConditionModel`, `AutoencoderKL`, `DDIMScheduler` |
| einops | `gla_model/gla.py:3` |
| omegaconf | `inject_gla_into_tf2dmodel` (`PixelHacker.py:398`); `load_cfg` dict branch |
| PyYAML | `load_cfg` string-path branch (`utils.py:9-11`) |
| Pillow | image / mask I/O |
| opencv-python | `pipeline.py:14`, `dataset.py:9` (`cv2.dilate` / `morphologyEx`) |
| numpy | pipeline / dataset |
| **flash-linear-attention (`fla`)** | **`gla_model/gla.py:16-18` — required to import the model class** |
| **triton** | **pinned in `requirements.txt`; required by `fla` GLA ops (Phase 4 MEASURED)** |
| transformers | imported by `gla.py:9` (`transformers.cache_utils`) |

`LCGModel` (`utils.py:18-36`) wraps the UNet with `nn.Embedding(20, encoder_hid_dim=3072)`. Inference uses learned category IDs, **not** a text encoder. No tokenizer is called on the denoise path.

### Listed but not imported by the inference path

| Package | `requirements.txt` | Import in `*.py` |
|---|---|---|
| **`flash-attn==2.5.8`** | yes | **none** — `rg flash_attn` hits only `requirements.txt:5` |
| lightning / pytorch-lightning | yes | none |
| lpips, tensorboard, scikit-learn, scipy, toml, … | yes | none on the infer entry point |

**VERIFIED FROM SOURCE:** `flash-attn` is a requirements-file entry, not an inference import. It is **not** the CUDA-only component that blocks local MPS. **NOT VALIDATED:** whether some optional training path imports it.

---

## B. Exact CUDA-only component(s)

**VERIFIED FROM SOURCE** plus **MEASURED** (Phase 4, same host).

1. **`fla` GLA operators — the blocking component.**  
   `gla_model/gla.py:16` imports `chunk_gla`, `fused_chunk_gla`, `fused_recurrent_gla` at **module load**. `DEFAULT_GLA_CONFIG` sets `mode = 'chunk'` (`gla.py:32`), so the live operator is `chunk_gla`. Those ops are Triton/CUDA kernels from `flash-linear-attention==0.3.2` (`requirements.txt:4`).  
   Phase 4 **MEASURED** on this M3 Pro: `fla` / Triton **cannot be installed** (no Apple Silicon wheels). That measurement is about the package, not about Moebius specifically, and it applies here because PixelHacker imports the same `fla.ops.gla` symbols.

2. **`triton==3.2.0`** (`requirements.txt:30`). Transitive hard requirement of `fla` ops. Same Phase 4 measurement: no macOS / ARM64 wheels from the official channel.

3. **Official device selection is CUDA-or-CPU, never MPS.**  
   `infer_pixelhacker.py:31`: `device = "cuda:0" if torch.cuda.is_available() else "cpu"`.  
   `pipeline.py:84` defaults `device='cuda'`.  
   `rg mps` over `*.py` is empty.

4. **`flash-attn==2.5.8`** is CUDA-compiled in general, but it is **not on the inference import graph** (see §A). Do not cite it as the reason LOCAL_MPS fails.

5. **`torch==2.3.0`** in `requirements.txt` is the upstream CUDA-era pin. That wheel is not the MPS blocker; `fla` is.

There is no upstream-supported “disable GLA / use vanilla attention” flag on the inference path. `use_gated_cross_attn` defaults to `False` (`PixelHacker.py:81`), which only controls whether **cross**-attention (`attn2`) is also replaced. Self-attention GLA (`attn1`) is **unconditional**.

---

## C. Why GLA cannot be removed without changing the research model

**VERIFIED FROM SOURCE.**

PixelHacker is not “a UNet plus an optional GLA teacher.” Construction **is** GLA injection:

`gla_model/PixelHacker.py:397-408` (verbatim structure):

```python
def inject_gla_into_tf2dmodel(tf2dmodel, use_gated_cross_attn=False):
    # ...
    for tfblock in tf2dmodel.transformer_blocks:
        glattn = GatedLinearAttention(**new_config)
        tfblock.attn1 = glattn
```

Called for every `CrossAttnDownBlock2D`, the mid `UNetMidBlock2DCrossAttn`, and every `CrossAttnUpBlock2D` (`PixelHacker.py:134-151`). After injection, `initialize_weights()` Xavier-initialises **all** `nn.Linear` modules, including the newly created GLA projections (`PixelHacker.py:156-163`). Trained GLA parameters then arrive only via `load_state_dict`.

Consequences:

| Proposed workaround | Why it is invalid |
|---|---|
| Moebius-style `sys.modules['model_lib']` surrogate | Wrong package. PixelHacker’s live class **is** `gla_model.PixelHacker`, which imports `gla.py`, which imports `fla`. Skipping that import skips the architecture. |
| Stub `fla` in `sys.modules` | `GatedLinearAttention` **is** `fla.models.gla.modeling_gla.GatedLinearAttention` (`gla.py:17, 26-28`). A stub would have to fake the real module, including `chunk_gla`. That is a reimplementation of the research operator. Forbidden. |
| Leave vanilla `attn1` in place | The published model and the published weights are GLA-UNet, not stock `UNet2DConditionModel`. |
| `load_state_dict(..., strict=False)` | Default load is `strict=True` (`infer_pixelhacker.py:49` — no `strict=` argument). `strict=False` can accept a checkpoint that **omits GLA keys**, leaving Xavier-random GLA modules in the forward pass. The result would not be PixelHacker. Unacceptable. |

Contrast with Moebius (Phase 4, **MEASURED**): GLA lives only on the **teacher** import (`model_lib/__init__.py:6`). The student architecture does not contain GLA, so isolating the teacher does not change student inference. PixelHacker **is** that teacher architecture. Isolation cannot be reused.

---

## D. Whether CPU execution is technically possible from source

**VERIFIED FROM SOURCE** (device string) vs **NOT VALIDATED** (actual CPU run).

`infer_pixelhacker.py:31` will select `"cpu"` when `torch.cuda.is_available()` is false. That is a **placement fallback**, not a GLA CPU kernel. The GLA forward still calls `fla.ops.gla` (`mode='chunk'`).

What this does **not** establish:

- that `chunk_gla` has a working PyTorch-CPU implementation in `flash-linear-attention==0.3.2`
- that the model can be imported on this Mac (it cannot, until `fla` exists — Phase 4 MEASURED)
- any CPU latency or memory figure

**No CPU inference was run. No CPU benchmark is claimed.**

---

## E. Whether CPU is a practical PixelForge target

**ESTIMATED**, with the Phase 4 `fla` install failure as **MEASURED** context.

No. Reasons:

1. On this host, the model class cannot be imported without `fla`, and `fla` is not installable (Phase 4).
2. Even on a machine where `fla` imports, the research kernels are Triton/CUDA. A silent CPU walk of those ops is not a supported PixelForge path under `docs/ENVIRONMENT_PLAN.md` §7 (algorithm must not be rewritten; CPU is only for models that actually run on CPU within acceptable limits).
3. The UNet checkpoint is **3 449 345 440 B** (~3.21 GiB) plus VAE **167 394 306 B** (HF API, 2026-08-27). A 0.8B diffusion UNet at 512×512, 20 steps, CFG batch-doubling, on CPU, is not a credible interactive backend. That last sentence is **ESTIMATED**; it is not a measurement.

Classification `CPU` is therefore rejected.

---

## F. Minimum checkpoint set for one inference

**VERIFIED FROM SOURCE** (which files the entry point loads) and **VERIFIED FROM SOURCE** (HF file listing / sizes). **Not downloaded this phase.**

Default weight path: `infer_pixelhacker.py:36` → `weight/ft_places2/diffusion_pytorch_model.bin`.  
VAE: `config/PixelHacker_sdvae_f8d4.yaml` `vae.model_dir: vae` + `utils.py:65-68`.

| Role | Path in upstream layout | HF path | Size (bytes) |
|---|---|---|---|
| **Required UNet** | `weight/ft_places2/diffusion_pytorch_model.bin` | `hustvl/PixelHacker/ft_places2/diffusion_pytorch_model.bin` | **3 449 345 440** |
| **Required VAE config** | `vae/config.json` | `hustvl/PixelHacker/vae/config.json` | 788 |
| **Required VAE weights** | `vae/diffusion_pytorch_model.bin` | `hustvl/PixelHacker/vae/diffusion_pytorch_model.bin` | **167 394 306** |

**Minimum set = those three files.** Combined payload **3 616 740 534 B (~3.37 GiB)**.

Not required for the default gate: `pretrained/`, `ft_celebahq/`, `ft_ffhq/`. Each of `pretrained/` and `ft_places2/` is the same 3 449 345 440 B class of file (HF API). README states each variant is “only 0.8B params” (`README.md:44`) — consistent with ~3.21 GiB float32.

PixelForge already holds a copy of the **VAE** under `checkpoints/moebius/vae/` from Phase 4 (local size 167 394 306 B, matching the HF listing). That does **not** satisfy PixelHacker inference: the UNet `ft_places2` file was **not** downloaded, by instruction.

---

## G. Checkpoint sources

**VERIFIED FROM SOURCE** (`README.md:82-103`) and HF API.

| Artifact | Source |
|---|---|
| Code | GitHub `hustvl/PixelHacker` @ `f5567db2871598aa178fe7a34c520dd478a0b41b` |
| Weights repo | `https://huggingface.co/hustvl/PixelHacker` |
| Weights revision recorded | `012fd343158936a265b8a0ee38a791a7a2841f45` (HF `sha`, also `REPOSITORIES.md`) |
| Default UNet | `https://huggingface.co/hustvl/PixelHacker/tree/main/ft_places2` |
| VAE | `https://huggingface.co/hustvl/PixelHacker/tree/main/vae` |

No download script exists in the GitHub tree. Layout is documented only as a directory tree in the README.

Note: `README.md:109-110` still shows `--weight weight/ft_places/diffusion_pytorch_model.bin` (missing `2`). The argparse default and the in-file comment were corrected to `ft_places2` (`infer_pixelhacker.py:36-37`). **Use `ft_places2`.**

---

## H. Weight / license considerations

**VERIFIED FROM SOURCE.**

| Work | License | Where |
|---|---|---|
| PixelHacker **code** | Apache-2.0 | `LICENSE` in the GitHub pin |
| PixelHacker **weights** | **MIT** | Hugging Face model card YAML `license: mit`; API `cardData.license = "mit"` |

This is a **split**: code Apache-2.0, weights MIT. Both are permissive OSI licenses. Unlike Moebius, PixelHacker’s GitHub README does **not** contain an explicit sentence that pretrained weights are Apache-2.0 and commercially usable. The weight license of record is the Hugging Face card (**MIT**), not the GitHub `LICENSE` file.

No PixelForge legal opinion is offered beyond recording those two published statements. Redistribution of weights must follow the HF MIT card.

---

## I. Recommended cloud-GPU environment

**ESTIMATED** environment shape for a future runtime gate. **NOT VALIDATED.** No instance was created.

| Item | Recommendation | Basis |
|---|---|---|
| OS / arch | Linux x86_64 | `fla` / Triton / CUDA wheels |
| Python | **3.10** | `README.md:76` `conda create -n pixelhacker python=3.10` |
| GPU | One NVIDIA GPU, **16 GB VRAM minimum, 24 GB preferred** | §K |
| CUDA toolkit matching the torch wheel | CUDA 11.8 or 12.1 class for `torch==2.3.0` | ESTIMATED from the published torch 2.3.0 wheel matrix — **NOT VALIDATED** |
| Isolation | Dedicated env `pixelforge-pixelhacker` (name reserved; **not created**) | `ENVIRONMENT_PLAN.md` §3 |

Do not share `pixelforge-moebius` or `pixelforge-sam2-v2` with this stack. Pins differ (`diffusers==0.30.2` vs Moebius 0.40.0; `torch==2.3.0` vs 2.13.0).

**Pin conflict to resolve at install time, not on paper:** `requirements.txt` lists `torch==2.3.0` and `triton==3.2.0` together. Public FLA/Triton notes pair Triton 3.2 with PyTorch **2.6**, not 2.3. Whether `pip install -r requirements.txt` succeeds as written is **NOT VALIDATED**. A future cloud setup script must record the **resolved** freeze, not only the upstream file.

Provider, region, and instance type are **not** chosen here. Credentials must stay in environment variables (`ENVIRONMENT_PLAN.md` §7).

---

## J. Required CUDA / PyTorch dependencies

**VERIFIED FROM SOURCE** (`requirements.txt`, `README.md:69-80`) for the **intended** cloud install. **NOT VALIDATED** by installation.

Minimum set for the **inference** path (not the entire requirements file):

| Pin | Role |
|---|---|
| `torch==2.3.0` (CUDA wheel) | tensors, UNet, VAE |
| `torchvision` (unpinned in file) | import of `transforms` |
| `diffusers==0.30.2` | UNet / VAE / DDIM |
| `transformers==4.40.0` | `gla.py` import |
| `flash-linear-attention==0.3.2` | GLA |
| `triton` (file says `3.2.0`; see pin conflict) | `fla` kernels |
| `einops==0.7.0` | `gla.py` |
| `omegaconf==2.3.0` | GLA inject + cfg |
| `accelerate==0.34.0` | listed; typical diffusers companion — **not traced as a direct infer import** |
| `safetensors==0.4.3` | listed; VAE load may use pickle `.bin` instead |
| opencv-python, numpy, PyYAML, Pillow, tqdm | I/O and loop |

Explicitly **not** required to *import* the infer path: `flash-attn`. It may still be pulled if someone installs the full `requirements.txt`. A future setup script should install the inference subset first.

`torch==2.7.1+cu130` (Moebius upstream pin) is **irrelevant** here. PixelHacker’s pin is `2.3.0`.

---

## K. Reasonable minimum VRAM

**ESTIMATED. NOT VALIDATED. No GPU run.**

| Term | Figure | Grade |
|---|---|---|
| UNet file size | 3.21 GiB | VERIFIED FROM SOURCE (HF size) |
| VAE file size | 0.16 GiB | VERIFIED FROM SOURCE (HF size); Phase 4 local file matches |
| Parameter count (README) | “0.8B” | VERIFIED FROM SOURCE (claim); 3.21 GiB / 4 ≈ **0.86×10⁹** scalars if float32 |
| Pipeline dtype at entry | `torch.float` (float32) | VERIFIED FROM SOURCE (`infer_pixelhacker.py:65`); pipeline signature default is float16 but the entry point overrides |
| CFG | `guidance_scale=4.5` ≠ 1 → tensors concatenated ×2 | VERIFIED FROM SOURCE (`pipeline.py:53-58`, infer defaults) |
| Resolution / steps | 512 / 20 | VERIFIED FROM SOURCE |

A float32 ~0.86B UNet plus VAE plus CFG-doubled 512×512 activations will not fit in 8 GB with comfortable headroom. **12 GB is a tight estimated floor; 16 GB is the reasonable minimum to attempt a first cloud smoke test; 24 GB is the comfortable default.**

This is **not** a measurement. OOM on a 16 GB card would not contradict this document; it would be the first runtime datum.

**No cloud latency is recorded or invented.**

---

## Mask contract

**VERIFIED FROM SOURCE.** Target chain: SAM 2 boolean H×W → PixelHacker mask.

### Input image

| Fact | Evidence |
|---|---|
| PIL **RGB** | `dataset.py:134` `Image.open(...).convert("RGB")` |
| Dataset resize | bicubic to `resolution` (default 512) if not already square (`dataset.py:135-136`) |
| Entry-point resize | `image.resize((img_size, img_size))` again (`infer_pixelhacker.py:76`); `img_size` from UNet `sample_size * vae_ds_ratio`, asserted equal to yaml `data.image_size` (512) |
| Pipeline geometry | scale so the **short side** is `image_size`, then floor to a multiple of 64 (`pipeline.py:126-136`) |

### Mask format

| Fact | Evidence |
|---|---|
| PIL **`"L"`** grayscale | `dataset.py:124` `.convert("L")` |
| Dataset resize | **NEAREST** to 512 (`dataset.py:129`) |
| Binarisation | `pipeline.py:119-120` threshold `255/2`: `point(lambda x: 0 if x < threshold else 255, 'L')` |
| Tensor mask | `mask = np.asarray(input_mask)/255.` then `where(mask >= 0.5, 1, 0)` (`pipeline.py:176-180`, `161-163`) |

### Mask polarity

**WHITE (255) = region to INPAINT; BLACK (0) = region to KEEP.**

- `masked_image = image * (1 - mask)` (`pipeline.py:165`) — content is **zeroed** where the mask is 1 (white).
- Same arithmetic as Moebius Phase 4. Independent of SAM 2.

Default `mask_dilate_kernel_size=0` → `mask_dilate` is a no-op (`pipeline.py:25-29`, infer `__call__` default).

Default `paste=False` (`infer_pixelhacker.py:86`) — unlike Moebius’s upstream `--pst` default True. Adapter authors must not silently copy Moebius paste behaviour.

### SAM 2 → PixelHacker

SAM 2 produces a boolean H×W array (Phase 3). Conversion:

```
mask_u8 = (mask_bool.astype(np.uint8)) * 255   # WHITE = inpaint
PIL Image mode "L"
```

That is the **same** conversion already recorded for Moebius. **Straightforward adapter conversion. No research-algorithm change.**

Caveats (not blockers):

1. PixelHacker’s pipeline may letterbox-scale to a 64-multiple; SAM 2 masks should be resized **NEAREST** with the same geometry the image uses.
2. `paste=False` by default, so unmasked pixels in the **output** are the decoded image, not a guaranteed copy of the input. Callers that need exact preservation must set `paste=True` explicitly.
3. End-to-end SAM 2 mask → PixelHacker output is **NOT VALIDATED** (no PixelHacker run).

---

## LOCAL_MPS conclusion — FAIL

A genuine local MPS path would require all of:

1. `fla` / Triton importable on macOS arm64, **or** an upstream non-GLA attention implementation;
2. an MPS device in the official entry point (or a placement-only change that still runs GLA on Metal);
3. GLA ops that execute on MPS.

None exist in the pinned tree. Phase 4 already **MEASURED** that (1) fails on this host. (2) is absent in source. Satisfying (1) by rewriting GLA in Metal / MLX would be a **new research operator**, which `ENVIRONMENT_PLAN.md` §7 forbids.

**LOCAL_MPS is FAIL**, not PARTIAL. The architecture was inspected; the missing piece is not “we did not try MPS” — it is that MPS cannot host this architecture without changing it.

No successful local MPS PixelHacker inference has been demonstrated. This phase does not add one.

---

## CLOUD_GPU conclusion — CONDITIONAL / NOT YET RUNTIME-VALIDATED

CUDA is the **documented** inference device. The dependency chain is closed in source. Placement `CLOUD_GPU` is the only classification that preserves the research model.

It is **CONDITIONAL** because:

- the exact `requirements.txt` freeze (especially `torch==2.3.0` + `triton==3.2.0`) is untested;
- `flash-attn` is in the file but unused by infer — install policy is not yet scripted;
- **no cloud process has loaded weights or produced an image.**

A later phase that actually runs PixelHacker on NVIDIA hardware may promote this to a runtime `PASS` / `CONDITIONAL` **execution** verdict. Until then: **classified, not validated.**

---

## Adapter boundary (not implemented)

Phase 6 must not begin from this document. The following is the **future** seam only.

`PixelHackerAdapter` hides cloud mechanics from the pipeline. Conceptually it exposes:

| Surface | Role |
|---|---|
| `availability` | whether the backend process/endpoint is reachable |
| `backend_type` | `CLOUD_GPU` (this classification) |
| `model_metadata` | pin SHA, checkpoint id (`ft_places2`), dtype, image size, license (code Apache-2.0 / weights MIT) |
| `image` in | PIL RGB (or array convertible to it) |
| `mask` in | boolean H×W or `'L'` 0/255; adapter performs the conversion in §Mask |
| `inference_parameters` | steps, guidance, strength, paste, seed — defaults from upstream, overridable |
| `output_image` | PIL RGB |
| `latency` | measured round-trip; **never a hardcoded estimate** |
| `memory` | reported by the worker if available; else `null` |
| `backend_status` | idle / loading / running / error, plus last failure class |

The rest of PixelForge talks only to this contract. Instance type, SSH, HTTP, CUDA, and `fla` stay inside the adapter. **This class does not exist in the tree yet.**

Local MVP inpainting remains **Moebius** (`LOCAL_MPS` / `CONDITIONAL`). PixelHacker is a cloud-capable second backend, not a replacement for the validated local path.

---

## What this phase did not do

- Download PixelHacker UNet weights
- Install PixelHacker Python dependencies
- Modify `research/upstream/PixelHacker`
- Create `pixelforge-pixelhacker`
- Deploy any cloud GPU
- Implement `PixelHackerAdapter`
- Run inference on CUDA, MPS, or CPU
- Rewrite Phase 3 or Phase 4 measurements

---

## Evidence index

| ID | Claim | Grade |
|---|---|---|
| A | Inference import chain through GLA / `fla` | VERIFIED FROM SOURCE |
| B | `fla` + Triton are the CUDA blockers; no MPS string | VERIFIED FROM SOURCE + MEASURED (Phase 4 install) |
| C | GLA replaces `attn1`; isolation / `strict=False` invalid | VERIFIED FROM SOURCE |
| D | Entry point can name `cpu` | VERIFIED FROM SOURCE |
| D′ | CPU GLA actually runs | **NOT VALIDATED** |
| E | CPU not a practical PixelForge target | ESTIMATED |
| F/G | Three-file minimum set and HF sizes | VERIFIED FROM SOURCE (HF API 2026-08-27) |
| H | Code Apache-2.0, weights MIT | VERIFIED FROM SOURCE |
| I/J | Cloud env / CUDA pins | VERIFIED FROM SOURCE (intended) + ESTIMATED (compat) + **NOT VALIDATED** (install) |
| K | VRAM 16 GB reasonable minimum | ESTIMATED |
| Mask | RGB + `'L'` + white-inpaint; SAM 2 conversion straightforward | VERIFIED FROM SOURCE |
| Runtime | Any successful PixelHacker image | **NOT VALIDATED** |
| Latency | Any cloud or CPU millisecond figure | **not claimed** |

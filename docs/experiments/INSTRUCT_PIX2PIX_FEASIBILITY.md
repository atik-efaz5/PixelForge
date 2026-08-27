# InstructPix2Pix Feasibility Classification

**Phase 12A — source audit and minimal integration plan for the pinned InstructPix2Pix implementation.**

This document answers: *can the pinned InstructPix2Pix checkout be added as a practical PixelForge instruction-editing backend without modifying upstream research code?*

**Answer (classification only): primary placement `CLOUD_GPU`. Local MPS on the Apple M3 Pro (18 GB unified) is not evidenced in source and is not validated.**

| Gate | Verdict |
|---|---|
| **LOCAL_MPS** | **NOT VALIDATED** — no MPS path in pinned inference sources; memory headroom doubtful |
| **CLOUD_GPU** | **CONDITIONAL / NOT YET RUNTIME-VALIDATED** — CUDA is hardcoded in the inference entry points |
| **CPU** | Not a practical PixelForge target (not benchmarked) |
| **UNAVAILABLE** | Not selected — a CUDA inference path exists in source |

No InstructPix2Pix weights were downloaded. No InstructPix2Pix environment was created. No inference was executed. Upstream `research/upstream/instruct-pix2pix` was not modified.

Evidence is labelled:

| Label | Meaning |
|---|---|
| **VERIFIED FROM SOURCE** | Read from commit `0dffd1eeb02611c35088462d1df88714ce2b52f4` |
| **VERIFIED FROM PUBLIC METADATA** | Named public registry (e.g. Hugging Face file listing), not executed here |
| **ESTIMATED** | Derived from architecture or third-party metadata; not measured on this host |
| **NOT VALIDATED** | Required for a future runtime gate; not claimed here |

---

## Repository

| Property | Value | Evidence |
|---|---|---|
| Path | `research/upstream/instruct-pix2pix` | VERIFIED FROM SOURCE |
| Remote | `https://github.com/timothybrooks/instruct-pix2pix.git` | `research/upstream/LOCKFILE.md` |
| **Pinned commit** | **`0dffd1eeb02611c35088462d1df88714ce2b52f4`** | LOCKFILE |
| Working tree | **clean** — not modified in this phase | policy |
| License (repo code) | **MIT (by verbatim text)** — `LICENSE` © 2023 Brooks/Holynski/Efros | VERIFIED FROM SOURCE |
| License (weights / SD lineage) | **CreativeML Open RAIL-M** — `stable_diffusion/LICENSE` | VERIFIED FROM SOURCE |

---

## 1. Inference path (VERIFIED FROM SOURCE)

### Primary entry points

| Script | Role |
|---|---|
| **`edit_cli.py`** | Minimal CLI: single image + instruction → edited PNG |
| **`edit_app.py`** | Gradio UI; same core `generate()` logic |
| `metrics/compute_metrics.py` | Evaluation harness (not a product inference path) |
| `main.py` | **Training** only (PyTorch Lightning, multi-GPU CUDA) |

**Recommended PixelForge reference path:** mirror `edit_cli.py` inside a future adapter (import upstream only inside `load()` / `infer()`).

### Call chain (`edit_cli.py`)

```
edit_cli.py:main()
  OmegaConf.load("configs/generate.yaml")
  load_model_from_config(config, ckpt, vae_ckpt=None)
    torch.load(ckpt) → instantiate_from_config → LatentDiffusion (ddpm_edit)
  model.eval().cuda()                                    # line 80 — hardcoded CUDA
  K.external.CompVisDenoiser(model)
  CFGDenoiser(model_wrap)                                # triple-batch CFG
  PIL RGB image → resize/fit to multiple of 64
  cond["c_crossattn"] = CLIP text embedding(instruction)
  cond["c_concat"]   = VAE.encode(input_image).mode()  # latent of source image
  uncond: empty text + zero image latent
  sigmas = model_wrap.get_sigmas(steps)                  # k-diffusion schedule
  z = randn * sigmas[0]
  K.sampling.sample_euler_ancestral(...)               # Euler ancestral sampler
  decode_first_stage(z) → clamp → uint8 PIL
```

**Scheduler:** Euler ancestral via **`k-diffusion`** (`K.sampling.sample_euler_ancestral`), not DDIM/PLMS from the vendored `stable_diffusion/ldm` samplers on this path.

**Default runtime parameters (`edit_cli.py`):**

| Parameter | Default |
|---|---|
| `resolution` | 512 (longest side scaled, dimensions snapped to multiples of 64) |
| `steps` | 100 |
| `cfg-text` | 7.5 |
| `cfg-image` | 1.5 |
| `config` | `configs/generate.yaml` |
| `ckpt` | `checkpoints/instruct-pix2pix-00-22000.ckpt` |
| `vae-ckpt` | `None` (VAE weights inside main checkpoint unless overridden) |

---

## 2. Model architecture (VERIFIED FROM SOURCE)

| Component | Implementation |
|---|---|
| Diffusion wrapper | `ldm.models.diffusion.ddpm_edit.LatentDiffusion` |
| Conditioning | **`hybrid`**: `c_crossattn` (CLIP text) + `c_concat` (encoded source image latents) |
| UNet | `UNetModel`, `in_channels=8`, `out_channels=4`, `model_channels=320` (`configs/generate.yaml`) |
| Text encoder | `FrozenCLIPEmbedder` (CLIP ViT-L/14, 768-d context) |
| First stage | `AutoencoderKL` (SD 1.x latent VAE, 4 channels, `scale_factor=0.18215`) |
| Training init | Fine-tuned from **Stable Diffusion v1.5** (`configs/train.yaml` `ckpt_path`) |

The UNet sees **8 input channels**: 4 noisy latents concatenated with 4 latent channels from the source image (`c_concat`). This is the InstructPix2Pix-specific modification over base SD 1.5.

**CFG denoiser (`CFGDenoiser`):** one forward pass batches **three** conditions (text+image, image-only, unconditional). This multiplies activation memory during each step.

---

## 3. Inference dependencies (VERIFIED FROM SOURCE)

Pinned `environment.yaml` (Conda, CUDA-era):

| Package | Pinned in `environment.yaml` |
|---|---|
| python | 3.8.5 |
| pytorch | 1.11.0 |
| cudatoolkit | 11.3 |
| torchvision | 0.12.0 |
| transformers | 4.19.2 |
| pytorch-lightning | 1.4.2 |
| omegaconf | 2.1.1 |
| einops | 0.3.0 |
| k-diffusion | git install |
| CLIP | openai/CLIP (editable) |
| taming-transformers | editable |
| diffusers | listed but **not used** by `edit_cli.py` |

**Packages reached by `edit_cli.py` / `edit_app.py` inference path:**

| Package | Use |
|---|---|
| torch | model, tensors, `autocast` |
| torchvision | indirect via LDM |
| PIL | image I/O |
| numpy | array bridge |
| einops | rearrange |
| omegaconf | config load |
| **k-diffusion** | sigma schedule + Euler ancestral sampler |
| **stable_diffusion/** (vendored LDM) | model, VAE, CLIP embedder |
| transformers | CLIP tokenizer/weights via `FrozenCLIPEmbedder` |
| openai CLIP | via cond stage config |

**CUDA-only in inference sources:**

| Location | Evidence |
|---|---|
| `edit_cli.py:80` | `model.eval().cuda()` |
| `edit_cli.py:98` | `autocast("cuda")` |
| `edit_app.py:110,160` | same pattern |
| `metrics/compute_metrics.py` | `.cuda()` throughout |

No `torch.mps` or device-agnostic `.to(device)` on the product inference path. **`stable_diffusion/scripts/img2img.py`** has a CUDA-or-CPU fallback, but that script is **not** the InstructPix2Pix edit path.

**NOT VALIDATED:** whether a PixelForge adapter could swap `.cuda()` for `mps` or `cpu` without upstream edits (adapter-side fork of the inference loop is allowed; upstream must stay read-only).

---

## 4. Checkpoint requirements (not downloaded)

### Inference (minimum)

| File | Source (pinned scripts) | Size |
|---|---|---|
| **`instruct-pix2pix-00-22000.ckpt`** | `scripts/download_checkpoints.sh` → `http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt` | **~7.7 GB** (VERIFIED FROM PUBLIC METADATA — Hugging Face `timbrooks/instruct-pix2pix`) |
| SHA-256 (HF listing) | `ffd280ddcfc8234e4d28b93641cb83169cebcb4d70998df9ee2eabb4d705374a` | not verified by download here |

`edit_cli.py` default `vae-ckpt=None` → **VAE weights are expected inside the main checkpoint** unless explicitly overridden.

### Training / dataset generation only (not required for inference)

| File | Source | Notes |
|---|---|---|
| `v1-5-pruned-emaonly.ckpt` | `scripts/download_pretrained_sd.sh` → Hugging Face `runwayml/stable-diffusion-v1-5` | SD 1.5 init for training |
| `vae-ft-mse-840000-ema-pruned.ckpt` | same script → `stabilityai/sd-vae-ft-mse-original` | optional improved VAE for dataset generation |
| GPT / image datasets | `scripts/download_data.sh` | not needed for inference |

### Base Stable Diffusion dependency

**VERIFIED FROM SOURCE:** model is a **fine-tune of Stable Diffusion v1.5** (`configs/train.yaml` `ckpt_path`, README §Training). Inference checkpoint is a merged Lightning `state_dict` containing UNet + VAE + CLIP conditioning stack.

---

## 5. Preprocessing & I/O contract (VERIFIED FROM SOURCE)

**Input**

- RGB image (`PIL.Image`, any size)
- Text **instruction** string (not a diffusion prompt for empty generation — an edit command, e.g. `"make the sky sunset"`)
- Resized with `ImageOps.fit` so both dimensions are multiples of **64**, longest side scaled to `--resolution` (default **512**)
- Normalized to `[-1, 1]` before VAE encode

**Output**

- Full-frame RGB edited image, same spatial size as the preprocessed input
- Saved as 8-bit PNG via PIL
- **Not** a mask, latent, or delta map

**Instruction conditioning**

- CLIP text embedding of the instruction (`c_crossattn`)
- Source image latent concatenated to UNet input (`c_concat`)
- Dual classifier-free guidance: `text_cfg_scale` and `image_cfg_scale` (defaults 7.5 and 1.5)

---

## 6. Mask support (VERIFIED FROM SOURCE)

**No mask-conditioned editing on the upstream inference path.**

- `rg mask` across `instruct-pix2pix/*.py` finds only attention masks inside the generic LDM transformer — not user region masks.
- `edit_dataset.py` training tuples are `(input_image, edited_image, instruction)` — no mask channel.
- `edit_cli.py` / `edit_app.py` accept **no mask argument**.

**PixelForge implication:** localized editing (`SAM2 mask + instruction`) is **not** supported by this upstream model. A future product workflow would need an **application-layer composite** (e.g. full-frame instruct edit + alpha blend with original using a SAM mask). That composite is **NOT VALIDATED** and is outside the pinned inference contract.

---

## 7. Device classification

| Placement | Verdict | Rationale |
|---|---|---|
| **CLOUD_GPU** | **Primary (CONDITIONAL / not runtime-validated)** | Inference hardcodes CUDA; README tested on **GPU with >18GB VRAM** |
| **LOCAL_MPS** | **NOT VALIDATED** | No MPS branch in `edit_cli.py`; CFG triple-batch + ~7.7 GB weights + 100 default steps |
| **CPU** | Impractical | No product CPU path; 100-step full UNet diffusion |
| **UNAVAILABLE** | No | Source clearly targets CUDA GPUs |

### Apple M3 Pro / 18 GB unified memory

| Factor | Assessment |
|---|---|
| README VRAM note | **>18 GB dedicated VRAM** tested (VERIFIED FROM SOURCE) — equals this host's **total** unified pool shared with OS |
| Checkpoint on disk | ~7.7 GB (VERIFIED FROM PUBLIC METADATA) |
| Runtime memory | **ESTIMATED** >> Moebius (~4.5 GB MPS measured): full SD-1.5-scale UNet, VAE, CLIP, **3× CFG batch**, 100 steps |
| MPS in source | **Absent** on inference path |
| Prior art on host | Moebius student (226 M params) validated `LOCAL_MPS`; InstructPix2Pix UNet is SD-1.5 scale (~860 M+ parameter class) |

**Conclusion:** A local MPS path is **not claimed**. Without runtime measurement, M3 Pro local inference is **implausible** for the pinned stack at default settings. Any future local experiment would require a dedicated env, device-ported inference loop, and reduced resolution/steps/precision — all **NOT VALIDATED**.

---

## 8. PixelForge fit

### Beside existing backends

| Backend | Role today | Relationship to InstructPix2Pix |
|---|---|---|
| **SAM 2** | Point/box segmentation → mask | Orthogonal — selection/refinement, not generation |
| **Grounding DINO** | Text → box → SAM mask (Phase 11) | Orthogonal — selection only |
| **Moebius** | Mask-conditioned inpainting (remove/fill) | **Different task** — needs mask, no instruction |
| **PixelHacker** | Mask-conditioned inpainting (cloud-classified) | Same split — mask inpaint vs global instruct edit |

### Supported upstream workflows

```
image + instruction  →  full edited image     ✅ VERIFIED (edit_cli.py)
SAM mask + instruction → localized edit        ❌ NOT in upstream (composite only, NOT VALIDATED)
```

### Proposed product roles (future, not implemented)

1. **Global instruct edit** — user provides instruction, receives full edited image (matches upstream).
2. **Select-then-edit (composite)** — SAM 2 mask for UI/refinement, Moebius for removal, InstructPix2Pix for stylistic/global edits; mask-guided instruct edit would be **PixelForge orchestration**, not upstream capability.

### Minimal integration plan (documentation only)

| Step | Action |
|---|---|
| 1 | **Source audit** (this document) — complete |
| 2 | Create isolated env `pixelforge-instruct-pix2pix` (Python 3.11 + modern torch); do **not** reuse `pixelforge-moebius` (diffusers 0.40 vs LDM/k-diffusion stack) or `pixelforge-sam2-v2` |
| 3 | Download `instruct-pix2pix-00-22000.ckpt` to `checkpoints/instruct_pix2pix/` (git-ignored) |
| 4 | Implement `InstructPix2PixAdapter` behind registry id `instruct_pix2pix` — vendored import inside `load()` only |
| 5 | Port `edit_cli.py` inference loop in adapter code (device-selectable wrapper; default `CLOUD_GPU`) |
| 6 | Add pipeline method `edit_by_instruction(image, instruction)` — **separate** from `inpaint(image, mask)` |
| 7 | Add API `POST /edit-by-instruction` — do not overload `/inpaint` |
| 8 | Runtime gate on **CUDA cloud** first; optional MPS experiment only if explicitly scheduled |
| 9 | Frontend: instruction field distinct from mask-based Generate |

**Alternative noted in README (not pinned repo):** Hugging Face **Diffusers** `StableDiffusionInstructPix2PixPipeline` (`timbrooks/instruct-pix2pix`). README claims lower GPU memory vs this LDM checkout. That is a **different dependency stack** — would require a **separate** audit before adoption. **NOT VALIDATED** for PixelForge.

---

## 9. Future adapter contract (design only)

```python
class InstructPix2PixAdapter(ModelAdapter):
    """Instruction-based full-image editing. Not mask inpainting."""

    backend_type: BackendType  # expected CLOUD_GPU until measured otherwise

    def infer(
        self,
        image: ImageArray,           # H×W×3 uint8 RGB
        instruction: str,
        /,
        *,
        mask: MaskArray | None = None,  # NOT USED by upstream; reserved for future composite workflows
        params: InstructPix2PixParams | None = None,
    ) -> InstructionEditResult: ...

@dataclass
class InstructPix2PixParams:
    num_steps: int | None = None          # default 100
    resolution: int | None = None         # default 512 (longest side)
    text_cfg_scale: float | None = None   # default 7.5
    image_cfg_scale: float | None = None  # default 1.5
    seed: int | None = None

@dataclass
class InstructionEditResult:
    result: ImageArray                    # H×W×3 uint8 RGB (may differ from input size after resize)
    latency_ms: float
    memory_mb: float | None
    model: str
    backend: BackendType
    metadata: dict[str, Any]              # steps, cfg scales, seed, output size, checkpoint path
```

**Separation from `InpaintingAdapter`:** mask inpainting (Moebius/PixelHacker) and instruction editing (InstructPix2Pix) should remain **distinct adapter roles** to avoid conflating contracts.

---

## 10. Licensing considerations (VERIFIED FROM SOURCE)

| Layer | License | PixelForge impact |
|---|---|---|
| InstructPix2Pix repo code | MIT (verbatim text in `LICENSE`) | Permissive for adapter wrapper code |
| Vendored `stable_diffusion/` | **CreativeML Open RAIL-M** | **Use-restricted** — governs model weights and derivatives |
| Checkpoints | Derived from SD 1.5 / RAIL-M lineage | Must comply with RAIL-M use restrictions for served outputs |
| Outputs | RAIL-M applies to model use | Product must review prohibited use cases (§5 of RAIL-M) before public deployment |

RAIL-M is the **strictest** license among the seven pinned research components for **deployment**, not merely redistribution.

---

## 11. Verified vs estimated vs not validated

| Topic | Status |
|---|---|
| Inference entry point (`edit_cli.py`) | **VERIFIED FROM SOURCE** |
| Hybrid CLIP + image-latent conditioning | **VERIFIED FROM SOURCE** |
| Euler ancestral + k-diffusion scheduler | **VERIFIED FROM SOURCE** |
| CUDA hardcoding | **VERIFIED FROM SOURCE** |
| No mask in inference API | **VERIFIED FROM SOURCE** |
| Checkpoint name and download URL | **VERIFIED FROM SOURCE** |
| Checkpoint size ~7.7 GB | **VERIFIED FROM PUBLIC METADATA** (HF) |
| README >18 GB VRAM note | **VERIFIED FROM SOURCE** |
| CLOUD_GPU classification | **ESTIMATED** from source evidence; **NOT RUNTIME-VALIDATED** |
| LOCAL_MPS on M3 Pro | **NOT VALIDATED** |
| Diffusers alternative viability | **NOT VALIDATED** |
| Mask-composite localized edit quality | **NOT VALIDATED** |
| Latency / VRAM on any device | **NOT VALIDATED** |
| Legal suitability for PixelForge product | **NOT VALIDATED** — RAIL-M review required |

---

## 12. Verdict

**Feasible as a PixelForge backend only behind explicit adapter + checkpoint + device work.** The pinned repository provides a clear single-image CLI inference path for **global instruction editing**. It does **not** support mask-localized instruction editing natively.

**Recommended classification:** **`CLOUD_GPU` (CONDITIONAL / NOT YET RUNTIME-VALIDATED)**.  
**Do not claim `LOCAL_MPS` on the M3 Pro** without a measured runtime gate.

No success claim is made. This phase is source audit and integration planning only.

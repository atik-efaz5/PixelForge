# Research Stack Plan

STATUS: **2 OF 7 RUNTIME-VALIDATED — SAM 2 (`LOCAL_MPS` / `PASS`) AND MOEBIUS (`LOCAL_MPS` / `CONDITIONAL`). GROUNDING DINO TEXT SELECTION IS VALIDATED (PHASE 11B). PIXELHACKER AND INSTRUCTPIX2PIX ARE CLASSIFIED `CLOUD_GPU` BUT NOT RUNTIME-VALIDATED.**

All seven research components have been cloned and pinned to exact commit SHAs (see [`research/upstream/LOCKFILE.md`](upstream/LOCKFILE.md)). **Acquisition is not validation.** Phase 3 ran the SAM 2 gate on real hardware and returned `PASS`: real segmentation inference executed on the Apple M3 Pro GPU via MPS. SAM 2 is therefore `VALIDATED` / `LOCAL_MPS`. Phase 4 has now taken Moebius through the full gate and returned `CONDITIONAL`: real generative inpainting executed on the same GPU, all 11 checks passed, but student inference on Apple Silicon requires a documented PixelForge-side import-isolation workaround that does not alter the research method. Moebius is therefore `VALIDATED (CONDITIONAL)` / `LOCAL_MPS`. Phase 5 classified PixelHacker from the pinned source **without** a runtime gate: `LOCAL_MPS` is `FAIL`; primary placement is `CLOUD_GPU`, **not yet runtime-validated**. The remaining four `Status` fields still read `NOT YET VALIDATED`.

Provenance (remote, commit SHA, license) is tracked separately and authoritatively in [`research/upstream/REPOSITORIES.md`](upstream/REPOSITORIES.md). This document covers intended role and validation state.

---

## No compatibility is claimed beyond what has been measured

Only SAM 2 and Moebius have been **runtime-measured**. PixelHacker has been **source-classified** (Phase 5) but has produced no image on any device in this project. For BrushNet, ControlNet, InstructPix2Pix, and Grounded-SAM, nothing here asserts that they run on this host, in any configuration.

- `Execution` is `TBD` where no dependency or device audit has been performed.
- `Local/cloud` is `TBD — NOT YET DETERMINED` because placement is an empirical outcome, not a design decision. Assigning it before measurement would be a guess.
- `Purpose` describes the **intended role** in PixelForge. It is a statement of our plan, not a claim about the component's behaviour on this machine.

## Validation gate

Every component must pass all seven stages, in order, before integration:

```
REPOSITORY AUDIT
  → DEPENDENCY AUDIT
    → DEVICE AUDIT
      → CHECKPOINT AUDIT
        → MINIMAL SMOKE TEST
          → PERFORMANCE TEST
            → INTEGRATION TEST
```

Only after all seven pass is a component classified `LOCAL_MPS`, `CLOUD_GPU`, `CPU`, or `UNAVAILABLE`.

A component that fails is recorded as failed, with the reason, in `REPOSITORIES.md`. Failures are results. Forcing a component past its gate — in particular by rewriting research code to make it run somewhere it was not designed to run — is prohibited.

---

## 1. SAM 2

**Purpose:** Promptable segmentation. Intended primary segmentation backend — click-based object selection, mask proposal, and mask refinement — and the refinement stage behind text-guided selection.

**Execution:** **Apple MPS (float32)** — measured, not assumed

**Local/cloud:** **`LOCAL_MPS`**

**Status:** **VALIDATED — 2026-08-24**

**Target phase:** Phase 3 — first reproducibility gate for the whole project. **Complete.**

**Notes:** A previous, now-deleted working tree reportedly ran SAM 2.1 Hiera-Tiny on Apple MPS. That tree was destroyed before it could be audited, so the report was treated as a hypothesis rather than prior art. Phase 3 re-established it independently, by execution.

Phase 2 surfaced one favourable piece of evidence: the pinned commit's own subject is an Apple MPS bug fix in `SAM2Base` (upstream #495). The Phase 3 measurement confirms that prior.

### Validated facts (Phase 3, 2026-08-24)

| Property | Value |
|---|---|
| Verdict | **`PASS`** — 11 of 11 criteria met |
| Commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` (verified twice) |
| Environment | `pixelforge-sam2-v2`, Python 3.11.15 |
| torch / torchvision / numpy | 2.13.0 / 0.28.0 / 2.4.6 |
| Device | `mps` — `model_device` = `mps:0`, no CPU fallback |
| Checkpoint | `sam2.1_hiera_tiny.pt`, 148.78 MiB, sha256 `7402e0d8…4be69` |
| Weights verified loaded | 470 tensors matched element-wise, 0 mismatched |
| Parameters | 38.96 M (documented 38.9 M) |
| Warm latency | **0.1508 s** total (`set_image` 0.1421 s + `predict` 0.0086 s) |
| Cold latency | 0.5710 s |
| Memory | peak RSS 789.41 MiB; MPS driver-allocated 1205.92 MiB of a 12288.02 MiB recommended ceiling |
| Gating mask | 8.2396 % area, non-empty, contains prompt point |

Warm `predict` is **23× faster** than cold (0.0086 s vs 0.2018 s) after Metal shader compilation, and warm latency is essentially resolution-independent (0.1508 s at 640×480 vs 0.1554 s at 1800×1200) because both are resized to the config's fixed 1024×1024. **Design consequence: encode once per image, then serve repeated clicks from the cached embedding.**

No IoU or reference-dependent metric is reported — no ground-truth masks exist. This is a reproducibility result, not an accuracy result.

Full record: [`docs/experiments/SAM2_MPS_VALIDATION.md`](../docs/experiments/SAM2_MPS_VALIDATION.md). Harness: [`tests/smoke/test_sam2_mps.py`](../tests/smoke/test_sam2_mps.py).

## 2. PixelHacker

**Purpose:** Generative inpainting. Project namesake and candidate inpainting backend (**Priority B**). Cloud-capable second backend; **not** the local MVP inpainting path (that is Moebius).

**Execution:** NVIDIA CUDA is the documented inference device. **Not executed in this project.**

**Local/cloud:** **`CLOUD_GPU`**

**Status:** **CLASSIFIED — 2026-08-27. NOT RUNTIME-VALIDATED.**

| Gate | Verdict |
|---|---|
| LOCAL_MPS | **FAIL** |
| CLOUD_GPU | **CONDITIONAL / NOT YET RUNTIME-VALIDATED** |
| CPU | not a practical PixelForge target (not benchmarked) |

**Target phase:** Phase 5. **Classification complete.** Adapter and cloud worker are **not** started.

**Notes:** Phase 5 answered whether a local path exists: **it does not**, unless the GLA UNet is rewritten. Acceptance of a *verified* cloud-GPU path remains open — source-feasible, never run.

### Classified facts (Phase 5, 2026-08-27)

| Property | Value |
|---|---|
| Commit | `f5567db2871598aa178fe7a34c520dd478a0b41b` (verified twice; tree clean) |
| Architecture | `UNet2DConditionModel` with **GLA replacing `attn1`** in every cross-attention block (`inject_gla_into_tf2dmodel`) |
| Blocking dependency | `flash-linear-attention` (`fla.ops.gla`) imported at `gla_model/gla.py:16` |
| Official device | `cuda:0` if CUDA else `cpu` — **no MPS path** |
| `flash-attn` | in `requirements.txt`, **not imported** by inference sources |
| Moebius isolation reusable? | **No** — GLA is the PixelHacker model, not a skippable teacher |
| `strict=False` | **unacceptable** — can leave randomly initialised GLA modules |
| Min checkpoints (not downloaded) | `ft_places2` UNet 3 449 345 440 B + VAE 167 394 306 B + `vae/config.json` |
| Weight license (HF) | **MIT**; GitHub **code** license Apache-2.0 |
| Environment | **not created** |
| Mask contract | PIL RGB + `'L'` mask, WHITE = inpaint; SAM 2 boolean → uint8 is a straightforward adapter conversion |

No cloud latency, no VRAM measurement, no successful PixelHacker image.

Full record: [`docs/experiments/PIXELHACKER_FEASIBILITY.md`](../docs/experiments/PIXELHACKER_FEASIBILITY.md).

## 3. Moebius

**Purpose:** Generative inpainting. Candidate inpainting backend (**Priority A**) — first to be evaluated.

**Execution:** **Apple MPS (float32)** — measured, not assumed

**Local/cloud:** **`LOCAL_MPS`**

**Status:** **VALIDATED (`CONDITIONAL`) — 2026-08-24**

**Target phase:** Phase 4. **Complete.**

**Notes:** Evaluated ahead of the other inpainting backends. Acceptance required producing a valid image **without modifying the research algorithm** — met. Training-only and CUDA-only dependencies were excluded where not required for inference, and none of the exclusions turned out to be needed.

### Validated facts (Phase 4, 2026-08-24)

| Property | Value |
|---|---|
| Verdict | **`CONDITIONAL`** — 11 of 11 criteria met |
| Commit | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` (verified twice, re-checked after the run) |
| License | `Apache-2.0` — `README.md:186`, covers code and pretrained weights |
| Environment | **`pixelforge-moebius`** (`/opt/anaconda3/envs/pixelforge-moebius`), Python 3.11.15 |
| torch / torchvision / numpy | 2.13.0 / 0.28.0 / 2.4.6 |
| diffusers / transformers | 0.40.0 / 5.15.1 |
| Device | **`mps`** — student and VAE both `mps:0`, no CPU fallback |
| Student parameters | **226 041 531** (226.04 M) |
| VAE parameters | 83 653 863 (83.65 M), `AutoencoderKL`, scaling 0.13025 |
| Student checkpoint | `ft_places2/diffusion_pytorch_model.bin`, 863.36 MiB, sha256 `6525afb8…6a09a` |
| VAE checkpoint | `vae/` (config.json + .bin), 159.64 MiB, from `hustvl/PixelHacker` |
| Weights verified loaded | **1203 tensors matched element-wise, 0 mismatched** |
| Model load / VAE load | 2.1335 s / 0.2174 s |
| Cold latency | 21.8105 s |
| **Warm latency** | **21.9121 s** (best of 3; mean 22.0682 s) at 512×512, 20 steps, batch 1 |
| Per denoising step | 1.0956 s |
| Memory | peak RSS 1589.23 MiB; **MPS driver-allocated 4498.22 MiB** of a 12288.02 MiB recommended ceiling (36.6 %) |
| Output | 512×512 RGB, finite, non-constant, 203 unique values, PNG lossless |
| Upstream modified | **No** — 0 changed files and 0 `__pycache__`, before and after |

**Why `CONDITIONAL` and not `PASS`.** `PASS` requires no compatibility workaround at all. On Apple Silicon the student cannot be imported without one: `model_lib/__init__.py:6` imports the CUDA-only PixelHacker **teacher**, whose chain reaches `flash-linear-attention` → Triton, which publishes no Apple Silicon wheels. Python executes a package's `__init__.py` before any submodule, so the *student* import dies on the *teacher's* dependency.

Resolved **without touching upstream**, by seeding `sys.modules['model_lib']` with a surrogate package that carries the real `__path__` and an empty body, then re-exporting exactly the 13 symbols `__init__.py` lines 2–3 provide and omitting only line 6 (the teacher). Verified teacher-free by execution, checked twice. The workaround **does not** modify upstream files, **does not** change the student architecture, **does not** stub or replace any research operation — it only controls import resolution. **This is a preserved, load-bearing requirement: it must survive into the adapter layer at Phase 6.**

Cold/warm ratio is **1.0×** — against SAM 2's 23× — because Moebius spends essentially all of its time in 20 sequential UNet denoising steps rather than one-off shader compilation. **Design consequence: pre-warming Moebius buys nothing, and latency scales linearly with step count at ~1.10 s/step.**

Also confirmed: the entry contract is PIL **RGB** image + PIL **`'L'`** grayscale mask with **WHITE (255) = inpaint, BLACK (0) = keep**, so the intended `SAM 2 boolean → uint8 0/255 → Moebius` chain is **correct**; and no text encoder, tokenizer or CLIP is on the inference path — `input_ids` index a learned `nn.Embedding(20, 3072)`.

No PSNR/SSIM/LPIPS/FID or any reference-dependent metric is reported — no ground truth exists for a synthetic scene. **This is a reproducibility result, not an accuracy result, and not a claim of production readiness:** ~21.9 s per 512×512 image is a substantial UX constraint, recorded as a measurement and not endorsed as acceptable.

Full record: [`docs/experiments/MOEBIUS_MPS_VALIDATION.md`](../docs/experiments/MOEBIUS_MPS_VALIDATION.md). Harness: [`tests/smoke/test_moebius_mps.py`](../tests/smoke/test_moebius_mps.py).

## 4. BrushNet

**Purpose:** Generative inpainting with dual-branch conditioning. Fallback inpainting backend (**Priority C**), should both Moebius and PixelHacker prove unviable.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Not scheduled. Contingent on Phase 4 and Phase 5 outcomes.

## 5. ControlNet

**Purpose:** Structural conditioning — structural preservation during editing.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 12. Not required for MVP.

## 6. InstructPix2Pix

**Purpose:** Instruction-based **global** image editing — `image + instruction → edited image` (e.g. *"make the sky sunset"*). Distinct from mask inpainting (Moebius / PixelHacker).

**Execution:** NVIDIA CUDA is hardcoded in the pinned inference entry points (`edit_cli.py`, `edit_app.py`). **Not executed in this project.**

**Local/cloud:** **`CLOUD_GPU`**

**Status:** **CLASSIFIED — 2026-08-27. NOT RUNTIME-VALIDATED.**

| Gate | Verdict |
|---|---|
| LOCAL_MPS | **NOT VALIDATED** — no MPS path in source; 18 GB unified memory unlikely at default settings |
| CLOUD_GPU | **CONDITIONAL / NOT YET RUNTIME-VALIDATED** |
| CPU | not a practical PixelForge target (not benchmarked) |

**Target phase:** Phase 12A classification complete. Implementation **not** started.

### Classified facts (Phase 12A, 2026-08-27)

| Property | Value |
|---|---|
| Commit | `0dffd1eeb02611c35088462d1df88714ce2b52f4` (pinned; tree clean) |
| Inference entry | `edit_cli.py` / `edit_app.py` |
| Architecture | `LatentDiffusion` hybrid: CLIP text + concatenated source-image latents; SD 1.5–scale UNet (`in_channels=8`) |
| Scheduler | Euler ancestral via **k-diffusion** |
| Default resolution / steps | 512 (64-multiple) / 100 |
| Mask support | **None** on upstream inference path |
| Inference checkpoint | `instruct-pix2pix-00-22000.ckpt` (~7.7 GB per HF metadata; not downloaded) |
| Base model lineage | Stable Diffusion v1.5 fine-tune |
| License (code) | MIT (verbatim text) |
| License (weights) | **CreativeML Open RAIL-M** via vendored `stable_diffusion/` |

Full record: [`docs/experiments/INSTRUCT_PIX2PIX_FEASIBILITY.md`](../docs/experiments/INSTRUCT_PIX2PIX_FEASIBILITY.md).

## 7. Grounded-Segment-Anything

**Purpose:** Open-vocabulary text-guided object grounding (including Grounding DINO) — natural-language selection such as "select the dog". PixelForge uses **Grounding DINO + SAM 2** directly (not the full Grounded-SAM monolith).

**Execution:** Grounding DINO **CPU** (isolated env); SAM 2 **LOCAL_MPS**

**Local/cloud:** Grounding **`CPU`**; selection path validated Phase 11B

**Status:** **GROUNDING PATH VALIDATED (PHASE 11B)** — full Grounded-SAM repo **NOT YET VALIDATED**

**Target phase:** Phase 11 complete for text selection; full repo remains unaudited for other features.

**Notes:** Phase 11B validated real Grounding DINO → SAM 2 box segmentation on this host. Two git submodules (`grounded-sam-osx`, `VISAM`) remain **uninitialized** by design. Full record: [`docs/experiments/GROUNDING_DINO_VALIDATION.md`](../docs/experiments/GROUNDING_DINO_VALIDATION.md).

---

## Status summary

| # | Component | Intended role | Priority | Target phase | Status |
|---|---|---|---|---|---|
| 1 | SAM 2 | Segmentation | Core / MVP | 3 | **VALIDATED — `LOCAL_MPS`** |
| 2 | PixelHacker | Inpainting | B | 5 | **CLASSIFIED `CLOUD_GPU` — `LOCAL_MPS: FAIL`; not runtime-validated** |
| 3 | Moebius | Inpainting | A | 4 | **VALIDATED (`CONDITIONAL`) — `LOCAL_MPS`** |
| 4 | BrushNet | Inpainting | C | contingent | NOT YET VALIDATED |
| 5 | ControlNet | Structural conditioning | Later | 12 | NOT YET VALIDATED |
| 6 | InstructPix2Pix | Instruction editing | Later | 12A | **CLASSIFIED `CLOUD_GPU` — not runtime-validated** |
| 7 | Grounded-Segment-Anything | Text-guided grounding | Later | 11 | **Grounding DINO path validated (11B); full repo not validated** |

**2 of 7 runtime-validated (SAM 2, Moebius). 4 of 7 classified or partially validated. 7 of 7 acquired and pinned.**

## MVP dependency

The MVP requires exactly **two** validated components — one segmentation backend and one inpainting backend:

```
UPLOAD → CLICK → SAM 2 SEGMENTATION → EDITABLE MASK
  → ONE VALIDATED INPAINTING BACKEND → RESULT
    → BEFORE / MASK / AFTER → PERFORMANCE METADATA
```

**Both MVP backends are now runtime-validated: SAM 2 (Phase 3, `PASS`) and Moebius (Phase 4, `CONDITIONAL`).** Moebius is the local inpainting backend, subject to preserving the documented import-isolation strategy. PixelHacker (Phase 5) is a **cloud-classified** second backend (`CLOUD_GPU`, `LOCAL_MPS: FAIL`) and is **not** an MVP blocker. The remaining four components are explicitly **not** MVP blockers and must not be started before the MVP path works end to end.

Two constraints the MVP design must carry:

- **The import isolation is load-bearing for Moebius.** Any adapter that imports `model_lib` before installing the surrogate will fail on `fla`. This must be preserved at Phase 6, not rediscovered. It does **not** make PixelHacker runnable on MPS: PixelHacker’s forward pass *is* `fla` GLA.
- **Latency is dominated by inpainting, not segmentation.** SAM 2 warm is 0.1508 s; Moebius warm is 21.9121 s — roughly **145×** more. The end-to-end warm path is therefore ~22 s, essentially all of it Moebius, and pre-warming does not help (cold/warm ratio 1.0×). Step count at ~1.10 s/step is the only obvious lever, and its quality cost is unmeasured.
- **PixelHacker is not wired and not run.** A future `PixelHackerAdapter` should expose availability, backend type `CLOUD_GPU`, metadata, image/mask I/O, parameters, output, latency, memory, and backend status, and hide CUDA/cloud details. It is **not implemented**.

Neither SAM 2 nor Moebius has been integrated yet: the SAM 2 → Moebius handoff has been verified only at the level of the mask contract (RGB + `'L'` mask, WHITE = inpaint), not end to end with a real SAM 2 mask. The same mask conversion applies to PixelHacker (Phase 5 source audit). That integration test remains Phase 6 for the **local** path.

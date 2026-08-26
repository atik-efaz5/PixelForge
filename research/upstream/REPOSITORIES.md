# Upstream Research Repository Registry

**Authoritative provenance record for every third-party research repository used by PixelForge.**

STATUS: **PHASE 1 — NO REPOSITORIES ACQUIRED**

Acquisition date: **TBD — NOT YET ACQUIRED** for all entries below.

Machine-readable pins: [`LOCKFILE.md`](LOCKFILE.md). Validation state and intended roles: [`../RESEARCH_STACK.md`](../RESEARCH_STACK.md).

---

## Policy

1. Everything under `research/upstream/` is an **external dependency** and is **READ-ONLY**. PixelForge code never modifies upstream research code; integration happens through adapters in `models/adapters/`.
2. Upstream clones are **not committed** to this repository (`.gitignore`: `research/upstream/*/`). Each carries its own `.git` history and its own license. Only `REPOSITORIES.md` and `LOCKFILE.md` are tracked.
3. **Commit SHAs are never invented.** A SHA is recorded only after being read from an actual local clone.
4. **Licenses are never assumed.** Recorded only after reading the license file *in the clone*.
5. Model weights are never committed. `Checkpoint source` records provenance so weights can be re-acquired later.
6. Classification as `LOCAL_MPS` / `CLOUD_GPU` / `CPU` / `UNAVAILABLE` happens only after the full validation gate.

### Phase 1 scope boundary

Phase 1 performs **registry scaffolding only**. No repository has been cloned, no dependency installed, no environment created, no checkpoint downloaded, and no upstream file modified. Every `Commit SHA`, `License`, `Branch`, and `Acquisition date` field therefore reads **TBD — NOT YET ACQUIRED** until Phase 2.

---

## 1. SAM 2

| Field | Value |
|---|---|
| **Name** | SAM 2 (Segment Anything Model 2) |
| **Remote** | https://github.com/facebookresearch/sam2.git |
| **Purpose** | Promptable visual segmentation in images and video. PixelForge role: primary segmentation backend — click-based selection, mask proposal, mask refinement. |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | Phase 3 target: SAM 2.1 Hiera-Tiny (smallest viable variant). |

## 2. PixelHacker

| Field | Value |
|---|---|
| **Name** | PixelHacker |
| **Remote** | https://github.com/hustvl/PixelHacker.git |
| **Purpose** | Image inpainting with structural and semantic consistency. PixelForge role: candidate inpainting backend, **Priority B**; project namesake. |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | Phase 5 determines local-vs-cloud. Teacher model for Moebius distillation. |

## 3. Moebius

| Field | Value |
|---|---|
| **Name** | Moebius |
| **Remote** | https://github.com/hustvl/Moebius.git |
| **Purpose** | Lightweight image inpainting framework. PixelForge role: candidate inpainting backend, **Priority A** — first to be evaluated. |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | VAE dependency may reference PixelHacker weights repository. |

## 4. BrushNet

| Field | Value |
|---|---|
| **Name** | BrushNet |
| **Remote** | https://github.com/TencentARC/BrushNet.git |
| **Purpose** | Plug-and-play image inpainting via decomposed dual-branch diffusion. PixelForge role: fallback inpainting backend, **Priority C**. |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | Activated only if Moebius and PixelHacker both fail. |

## 5. ControlNet

| Field | Value |
|---|---|
| **Name** | ControlNet |
| **Remote** | https://github.com/lllyasviel/ControlNet.git |
| **Purpose** | Adding conditional control to text-to-image diffusion models. PixelForge role: structural conditioning (Phase 12). |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | Deferred — not MVP. |

## 6. InstructPix2Pix

| Field | Value |
|---|---|
| **Name** | InstructPix2Pix |
| **Remote** | https://github.com/timothybrooks/instruct-pix2pix.git |
| **Purpose** | Instruction-based image editing from natural-language instructions. PixelForge role: instruction-based editing (Phase 12). |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | Deferred — not MVP. |

## 7. Grounded-Segment-Anything

| Field | Value |
|---|---|
| **Name** | Grounded-Segment-Anything (incl. Grounding DINO) |
| **Remote** | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |
| **Purpose** | Open-vocabulary detection + segmentation from text prompts. PixelForge role: text-guided selection (Phase 11). |
| **License** | TBD — NOT YET ACQUIRED |
| **Commit SHA** | TBD — NOT YET ACQUIRED |
| **Branch** | TBD — NOT YET ACQUIRED |
| **Acquisition date** | TBD — NOT YET ACQUIRED |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | TBD — NOT YET ACQUIRED |
| **Validation status** | NOT YET VALIDATED |
| **Notes** | May declare git submodules; initialization deferred to Phase 2 audit. |

---

## Summary

| # | Repository | Remote | Commit SHA | License | Validation status |
|---|---|---|---|---|---|
| 1 | SAM 2 | facebookresearch/sam2 | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 2 | PixelHacker | hustvl/PixelHacker | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 3 | Moebius | hustvl/Moebius | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 4 | BrushNet | TencentARC/BrushNet | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 5 | ControlNet | lllyasviel/ControlNet | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 6 | InstructPix2Pix | timothybrooks/instruct-pix2pix | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |
| 7 | Grounded-Segment-Anything | IDEA-Research/Grounded-Segment-Anything | TBD — NOT YET ACQUIRED | TBD — NOT YET ACQUIRED | NOT YET VALIDATED |

**0 of 7 acquired. 0 of 7 pinned. 0 of 7 validated. 0 weights downloaded. 0 environments created.**

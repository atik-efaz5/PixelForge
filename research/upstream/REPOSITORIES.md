# Upstream Research Repository Registry

**Authoritative provenance record for every third-party research repository used by PixelForge.**

STATUS: **PHASE 2 COMPLETE — ALL 7 REPOSITORIES ACQUIRED AND PINNED**

Acquisition date: **2026-08-27**. Every SHA below was read from the local clone via `git rev-parse HEAD` and verified twice. No SHA was inferred, guessed, or taken from a web API.

Machine-readable pins: [`LOCKFILE.md`](LOCKFILE.md). Validation state and intended roles: [`../RESEARCH_STACK.md`](../RESEARCH_STACK.md).

---

## Policy

1. Everything under `research/upstream/` is an **external dependency** and is **READ-ONLY**. PixelForge code never modifies upstream research code; integration happens through adapters in `models/adapters/`.
2. Upstream clones are **not committed** to this repository (`.gitignore`: `research/upstream/*/`). Each carries its own `.git` history and its own license. Only `REPOSITORIES.md` and `LOCKFILE.md` are tracked.
3. **Commit SHAs are never invented.** A SHA is recorded only after being read from an actual local clone.
4. **Licenses are never assumed.** Recorded only after reading the license file *in the clone*. Where a repository's own files do not clearly establish an identifier, the entry reads `REVIEW REQUIRED`.
5. Model weights are never committed. `Checkpoint source` records provenance so weights can be re-acquired and integrity-checked later.
6. Classification as `LOCAL_MPS` / `CLOUD_GPU` / `CPU` / `UNAVAILABLE` happens only after the full validation gate.

### Phase 2 scope boundary

Phase 2 performed **acquisition and provenance only**. No dependency was installed, no environment created, no checkpoint downloaded, no upstream file modified, no upstream setup script executed, and no submodule initialized. Every `Validation status` reads `NOT YET VALIDATED`. Device and dependency audits belong to Phase 3 onward.

### Verified: no weights present

A scan of all seven clones for `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, `*.onnx` returned **0 files**. No download script was executed.

---

## 1. SAM 2

| Field | Value |
|---|---|
| **Name** | SAM 2 (Segment Anything Model 2) |
| **Remote** | https://github.com/facebookresearch/sam2.git |
| **Commit SHA** | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:39+0600 |
| **License** | **Apache-2.0** — `LICENSE` (verbatim Apache License 2.0 header). Additionally `LICENSE_cctorch` = **BSD-3-Clause** covering a vendored third-party component. |
| **Purpose** | Promptable visual segmentation in images and video. Foundation model extending SAM to video by treating images as single-frame video. PixelForge role: primary segmentation backend — click-based selection, mask proposal, mask refinement. |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Documented in README and `checkpoints/download_ckpts.sh`. SAM 2.1 series (`092824/`): `sam2.1_hiera_tiny.pt`, `_small.pt`, `_base_plus.pt`, `_large.pt` from `dl.fbaipublicfiles.com/segment_anything_2/092824/`. Helper script present, **not executed**. |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared (no `.gitmodules`) |
| **Dependency files** | `pyproject.toml`, `setup.py` |
| **Disk usage** | 208 MB (`.git`: 145 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | HEAD commit subject is an explicit Apple MPS bug fix in `SAM2Base` (#495). Encouraging signal for Phase 3, but **not** evidence of execution on this host. |

## 2. PixelHacker

| Field | Value |
|---|---|
| **Name** | PixelHacker |
| **Remote** | https://github.com/hustvl/PixelHacker.git |
| **Commit SHA** | `f5567db2871598aa178fe7a34c520dd478a0b41b` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:39+0600 |
| **License** | **Apache-2.0** — `LICENSE` (verbatim Apache License 2.0). README badge concurs. |
| **Purpose** | Image inpainting with structural and semantic consistency (Latent Categories Guidance). Reports SOTA on Places2, CelebA-HQ, and FFHQ. PixelForge role: candidate inpainting backend, **Priority B**; project namesake. |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Hugging Face `hustvl/PixelHacker` — subtrees `pretrained/`, `ft_places2/`, `ft_celebahq/`, `ft_ffhq/`, and `vae/`. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `requirements.txt` |
| **Disk usage** | 223 MB (`.git`: 192 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Teacher model for Moebius distillation. Phase 5 determines local-vs-cloud. |

## 3. Moebius

| Field | Value |
|---|---|
| **Name** | Moebius |
| **Remote** | https://github.com/hustvl/Moebius.git |
| **Commit SHA** | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:39+0600 |
| **License** | **Apache-2.0 — covering both code and pretrained weights.** `LICENSE` (verbatim Apache 2.0). README §License states explicitly that code and pretrained model weights are released under Apache 2.0 and commercial use of weights is permitted. |
| **Purpose** | 0.2B-parameter lightweight image inpainting framework (ECCV'26). Uses adaptive multi-granularity distillation from PixelHacker (teacher). PixelForge role: candidate inpainting backend, **Priority A** — first to be evaluated. |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Hugging Face `hustvl/Moebius` — subtrees `pretrained/`, `ft_places2/`, `ft_celebahq/`, `ft_ffhq/`. **Additionally requires VAE from** `hustvl/PixelHacker/tree/main/vae`. Expected layout `./weight/vae` and `./weight/Moebius`. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `requirements.txt` |
| **Disk usage** | 165 MB (`.git`: 112 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Clearest weight-licensing position of the seven repos. VAE dependency on PixelHacker HF repo means Phase 4 cannot proceed on Moebius weights alone. |

## 4. BrushNet

| Field | Value |
|---|---|
| **Name** | BrushNet |
| **Remote** | https://github.com/TencentARC/BrushNet.git |
| **Commit SHA** | `0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:40+0600 |
| **License** | **Apache-2.0 for BrushNet itself** — `LICENSE` declares BrushNet is licensed under Apache 2.0 **except for third-party components listed below**. ⚠️ **Third-party components: REVIEW REQUIRED** — the exception clause is declared but the file contains no enumerated list after it. |
| **Purpose** | Plug-and-play image inpainting via decomposed dual-branch diffusion (ECCV 2024). PixelForge role: fallback inpainting backend, **Priority C**. |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Google Drive folder linked in README (`drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n`). Also references Hugging Face `TencentARC/BrushEdit/tree/main/brushnetX` (BrushNetX). Training data: `huggingface.co/datasets/random123123/BrushData`. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `pyproject.toml`, `setup.py`, `Makefile` |
| **Disk usage** | 78 MB (`.git`: 39 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Smallest clone of the seven. Activated only if Moebius (Phase 4) and PixelHacker (Phase 5) both fail. |

## 5. ControlNet

| Field | Value |
|---|---|
| **Name** | ControlNet |
| **Remote** | https://github.com/lllyasviel/ControlNet.git |
| **Commit SHA** | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:40+0600 |
| **License** | **Apache-2.0** — `LICENSE` (verbatim Apache License 2.0). |
| **Purpose** | Adding conditional control to text-to-image diffusion models (arXiv 2302.05543). PixelForge role: structural conditioning and structural preservation during editing (Phase 12). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Hugging Face `lllyasviel/ControlNet` (models in `ControlNet/models`, detectors in `ControlNet/annotator/ckpts` per README). **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `environment.yaml` (Conda) |
| **Disk usage** | 230 MB (`.git`: 129 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Oldest pin of the seven — HEAD dates 2023-09-09. README states this is ControlNet 1.0; ControlNet 1.1 exists in a separate repository. Deferred — not MVP. |

## 6. InstructPix2Pix

| Field | Value |
|---|---|
| **Name** | InstructPix2Pix |
| **Remote** | https://github.com/timothybrooks/instruct-pix2pix.git |
| **Commit SHA** | `0dffd1eeb02611c35088462d1df88714ce2b52f4` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:40+0600 |
| **License** | **MIT (by text) for the repository's own code** — `LICENSE` is the verbatim MIT permission text, © 2023 Timothy Brooks, Aleksander Holynski, Alexei A. Efros. The file never names "MIT" explicitly; identifier inferred from verbatim body text. ⚠️ **Weights and derived code: REVIEW REQUIRED.** The same file states portions derive from Stable Diffusion under CompVis/stable-diffusion, that further restrictions may apply, and directs the reader to `stable_diffusion/LICENSE` — which is **CreativeML Open RAIL-M**, a use-restricted license. |
| **Purpose** | Instruction-based image editing from natural-language instructions. Built on CompVis Stable Diffusion. PixelForge role: instruction-based editing (Phase 12). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | `instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt` (via `scripts/download_checkpoints.sh`). Base SD: `huggingface.co/runwayml/stable-diffusion-v1-5` and `huggingface.co/stabilityai/sd-vae-ft-mse-original` (via `scripts/download_pretrained_sd.sh`). **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `environment.yaml` (Conda) |
| **Disk usage** | 37 MB (`.git`: 18 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | README states instructions were tested on a GPU with >18GB VRAM. RAIL-M licensing on weights is the strictest of the seven. Deferred — not MVP. |

## 7. Grounded-Segment-Anything

| Field | Value |
|---|---|
| **Name** | Grounded-Segment-Anything (incl. Grounding DINO) |
| **Remote** | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |
| **Commit SHA** | `126abe633ffe333e16e4a0a4e946bc1003caf757` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-27T03:19:40+0600 |
| **License** | **Apache-2.0** — `LICENSE` (verbatim Apache License 2.0). ⚠️ Scope caveat: composite project composing several independently-licensed upstream models. Two declared submodules are **uninitialized** — their licenses were not inspected. |
| **Purpose** | Combines Grounding DINO (open-set detection) with Segment Anything to detect and segment anything from a text prompt. PixelForge role: text-guided / natural-language selection (Phase 11). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | `groundingdino_swint_ogc.pth` (GroundingDINO GitHub release v0.1.0-alpha); `sam_vit_h_4b8939.pth` (`dl.fbaipublicfiles.com`). Optional LLaVA and other weights referenced in README. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | ⚠️ **2 declared, both UNINITIALIZED** — `grounded-sam-osx` → `https://github.com/linjing7/grounded-sam-osx.git` @ `6688b036c7856a302f9315bb16864d66fb2cdade`; `VISAM` → `https://github.com/BingfengYan/VISAM` @ `d7c38233882ff9d34d5cbecb8495e175e4dffc8c`. `git submodule status` shows leading `-` on both. Submodule contents **not on disk**. |
| **Dependency files** | `requirements.txt`, `Makefile`, `Dockerfile` |
| **Disk usage** | 278 MB (`.git`: 160 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Historically requires CUDA-compiled extensions for Grounding DINO. README notes successor `Grounded-SAM-2` pairing Grounding DINO with SAM 2. Deferred. |

---

## Summary

| # | Repository | Commit SHA | Branch | License | Size | Tree |
|---|---|---|---|---|---|---|
| 1 | SAM 2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | main | Apache-2.0 (+BSD-3-Clause vendored) | 208 MB | clean |
| 2 | PixelHacker | `f5567db2871598aa178fe7a34c520dd478a0b41b` | main | Apache-2.0 | 223 MB | clean |
| 3 | Moebius | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` | main | Apache-2.0 (code **and** weights) | 165 MB | clean |
| 4 | BrushNet | `0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d` | main | Apache-2.0; third-party carve-out **REVIEW REQUIRED** | 78 MB | clean |
| 5 | ControlNet | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` | main | Apache-2.0 | 230 MB | clean |
| 6 | InstructPix2Pix | `0dffd1eeb02611c35088462d1df88714ce2b52f4` | main | MIT by text; weights **CreativeML Open RAIL-M — REVIEW REQUIRED** | 37 MB | clean |
| 7 | Grounded-Segment-Anything | `126abe633ffe333e16e4a0a4e946bc1003caf757` | main | Apache-2.0 (submodules uninspected) | 278 MB | clean |

**7 of 7 acquired. 7 of 7 pinned. 7 of 7 clean. 0 of 7 validated. 0 of 7 classified. 0 weights downloaded. 0 environments created.**

Total upstream disk usage: **1.2 GB**.

### Licensing attention required

| Repository | Issue |
|---|---|
| BrushNet | Declares a third-party exception to Apache-2.0 but does not enumerate the components. Unresolvable from repo files. |
| InstructPix2Pix | Own code MIT *by text only*. Checkpoints derive from Stable Diffusion under **CreativeML Open RAIL-M** — use-restricted. |
| Grounded-Segment-Anything | Composite project; two uninitialized submodules have uninspected licenses. |
| SAM 2 | Secondary `LICENSE_cctorch` (BSD-3-Clause) applies to a vendored component. |

Moebius is the only repository giving an unambiguous, permissive position on **weights** as well as code.

## Re-verification

```
cd /Users/atik/Projects/PixelForge/research/upstream
for r in sam2 PixelHacker Moebius BrushNet ControlNet instruct-pix2pix Grounded-Segment-Anything; do
  printf "%-28s %s  %s changed\n" "$r" "$(git -C $r rev-parse HEAD)" "$(git -C $r status --porcelain | wc -l | tr -d ' ')"
done
```

Compare against [`LOCKFILE.md`](LOCKFILE.md). Any mismatch means a working tree was altered or updated, which invalidates every result derived from it until re-recorded.

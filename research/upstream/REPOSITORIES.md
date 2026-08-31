# Upstream Research Repository Registry

**Authoritative provenance record for every third-party research repository used by PixelForge.**

STATUS: **PHASE 5 COMPLETE FOR CLASSIFIED MODELS — SAM 2 (`LOCAL_MPS` / `PASS`), MOEBIUS (`LOCAL_MPS` / `CONDITIONAL`), PIXELHACKER (`LOCAL_MPS` / `FAIL`, primary `CLOUD_GPU`). FOUR REMAINING REPOSITORIES NOT YET VALIDATED.**

Acquisition date: **2026-08-24**. Every SHA below was read from the local clone via `git rev-parse HEAD` and verified twice. No SHA was inferred, guessed, or taken from a web API.

Machine-readable pins: [`LOCKFILE.md`](LOCKFILE.md). Validation state and intended roles: [`../RESEARCH_STACK.md`](../RESEARCH_STACK.md). Runtime records: [`docs/experiments/SAM2_MPS_VALIDATION.md`](../../docs/experiments/SAM2_MPS_VALIDATION.md), [`docs/experiments/MOEBIUS_MPS_VALIDATION.md`](../../docs/experiments/MOEBIUS_MPS_VALIDATION.md), [`docs/experiments/PIXELHACKER_MPS_VALIDATION.md`](../../docs/experiments/PIXELHACKER_MPS_VALIDATION.md).

---

## Policy

1. Everything under `research/upstream/` is an **external dependency** and is **READ-ONLY**. PixelForge code never modifies upstream research code; integration happens through adapters in `models/adapters/`.
2. Upstream clones are **not committed** to this repository (`.gitignore`: `research/upstream/*/`). Each carries its own `.git` history and its own license. Only `REPOSITORIES.md` and `LOCKFILE.md` are tracked.
3. **Commit SHAs are never invented.** A SHA is recorded only after being read from an actual local clone.
4. **Licenses are never assumed.** Recorded only after reading the license file *in the clone*. Where a repository's own files do not clearly establish an identifier, the entry reads `License: REVIEW REQUIRED`.
5. Model weights are never committed. `Checkpoint source` records provenance so weights can be re-acquired and integrity-checked later.
6. Classification as `LOCAL_MPS` / `CLOUD_GPU` / `CPU` / `UNAVAILABLE` happens only after: repository audit → dependency audit → device audit → checkpoint audit → minimal smoke test → performance test → integration test.

### Phase 2 scope boundary

Phase 2 performed **acquisition and provenance only**. Phases 3–5 added runtime gates for SAM 2, Moebius, and PixelHacker respectively. Entries below retain acquisition facts; **Validation status** and **Environment** are updated only where a phase gate has completed.

### Verified: no weights present

A scan of all seven clones for `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, `*.onnx` returned **0 files**. `sam2/checkpoints/` contains only the upstream `download_ckpts.sh`, which was **not executed**.

---

## 1. SAM 2

| Field | Value |
|---|---|
| **Name** | SAM 2 (Segment Anything Model 2) |
| **Remote** | https://github.com/facebookresearch/sam2.git |
| **Commit SHA** | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:38:50+0600 |
| **License** | **Apache-2.0** — `LICENSE` (201 lines, verbatim Apache License 2.0 header). Additionally `LICENSE_cctorch` = **BSD-3-Clause** covering a vendored third-party component. |
| **Purpose** | Promptable visual segmentation in images and video. Foundation model extending SAM to video by treating images as single-frame video, with a streaming-memory architecture. PixelForge role: primary segmentation backend — click-based selection, mask proposal, mask refinement, and the refinement stage behind text-guided selection. |
| **Environment** | **`pixelforge-sam2-v2`** — Python 3.11.15, torch 2.13.0 |
| **Checkpoint source** | Documented in README, `dl.fbaipublicfiles.com`. Phase 3 gate: `sam2.1_hiera_tiny.pt` (148.78 MiB, present under `checkpoints/sam2/`). |
| **Validation status** | **VALIDATED — `LOCAL_MPS` / `PASS` (2026-08-24)** — [`docs/experiments/SAM2_MPS_VALIDATION.md`](../../docs/experiments/SAM2_MPS_VALIDATION.md) |
| **Submodules** | None declared (no `.gitmodules`) |
| **Dependency files** | `pyproject.toml`, `setup.py` |
| **Disk usage** | 208 MB (`.git`: 145 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Phase 3 **PASS** on Apple M3 Pro MPS — warm latency 0.1508 s, no CPU fallback. HEAD commit subject is an Apple MPS bug fix (#495); measurement confirms it. |

## 2. PixelHacker

| Field | Value |
|---|---|
| **Name** | PixelHacker |
| **Remote** | https://github.com/hustvl/PixelHacker.git |
| **Commit SHA** | `f5567db2871598aa178fe7a34c520dd478a0b41b` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:39:47+0600 |
| **License** | **Apache-2.0** — `LICENSE` (201 lines, verbatim Apache License 2.0). README badge concurs. |
| **Purpose** | Image inpainting with structural and semantic consistency. Reports SOTA on Places2, CelebA-HQ and FFHQ. Authors: Huazhong University of Science and Technology + VIVO AI Lab (arXiv 2504.20438). PixelForge role: candidate inpainting backend, **Priority B**; project namesake. |
| **Environment** | **not created** — Phase 5 gate used host MPS probe only; no `pixelforge-pixelhacker` |
| **Checkpoint source** | Hugging Face `hustvl/PixelHacker` — `ft_places2` UNet + `vae/`. **UNet not downloaded.** VAE copy may exist under `checkpoints/moebius/vae/` from Phase 4. |
| **Validation status** | **CLASSIFIED — `LOCAL_MPS` / `FAIL`, primary `CLOUD_GPU` (2026-08-31)** — [`docs/experiments/PIXELHACKER_MPS_VALIDATION.md`](../../docs/experiments/PIXELHACKER_MPS_VALIDATION.md), [`docs/experiments/PIXELHACKER_FEASIBILITY.md`](../../docs/experiments/PIXELHACKER_FEASIBILITY.md) |
| **Submodules** | None declared |
| **Dependency files** | `requirements.txt` |
| **Disk usage** | 223 MB (`.git`: 192 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Phase 5 gate: `fla`/GLA blocks import on Apple Silicon; upstream device line is CUDA-or-CPU only. Cloud CUDA path is source-feasible but **not runtime-validated**. Teacher model for Moebius distillation. HF weights revision `012fd343…` is not this GitHub SHA. |

## 3. Moebius

| Field | Value |
|---|---|
| **Name** | Moebius |
| **Remote** | https://github.com/hustvl/Moebius.git |
| **Commit SHA** | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:41:27+0600 |
| **License** | **Apache-2.0 — covering both code and pretrained weights.** `LICENSE` (199 lines, verbatim Apache 2.0). README §License states explicitly: *"Both the code and the pretrained model weights of Moebius are released under the Apache License 2.0 … Commercial use of the weights and the images produced with them is permitted."* |
| **Purpose** | 0.2B-parameter lightweight image inpainting framework claiming 10B-level performance (ECCV'26, arXiv 2606.19195). Uses adaptive multi-granularity distillation transferring representational capacity from PixelHacker (teacher) within latent space. PixelForge role: candidate inpainting backend, **Priority A** — first to be evaluated for local MPS. |
| **Environment** | **`pixelforge-moebius`** — Python 3.11.15, torch 2.13.0, diffusers 0.40.0 |
| **Checkpoint source** | Hugging Face `hustvl/Moebius` `ft_places2` + VAE from `hustvl/PixelHacker/vae`. Present under `checkpoints/moebius/`. |
| **Validation status** | **VALIDATED — `LOCAL_MPS` / `CONDITIONAL` (2026-08-24)** — [`docs/experiments/MOEBIUS_MPS_VALIDATION.md`](../../docs/experiments/MOEBIUS_MPS_VALIDATION.md) |
| **Submodules** | None declared |
| **Dependency files** | `requirements.txt` |
| **Disk usage** | 165 MB (`.git`: 113 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Phase 4 **CONDITIONAL** on Apple M3 Pro MPS — warm inpainting 21.9 s @ 512², student-only import surrogate required. Explicit Apache-2.0 license covering code and weights. |

## 4. BrushNet

| Field | Value |
|---|---|
| **Name** | BrushNet |
| **Remote** | https://github.com/TencentARC/BrushNet.git |
| **Commit SHA** | `0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:42:19+0600 |
| **License** | **Apache-2.0 for BrushNet itself** — `LICENSE` line 1–5: *"Tencent is pleased to support the open-source community… Copyright (C) 2024 THL A29 Limited, a Tencent company… BrushNet is licensed under the Apache License Version 2.0 **except for the third-party components listed below**."* ⚠️ **Third-party components: REVIEW REQUIRED** — the exception clause is declared but the file contains no enumerated list after it (210 lines, remainder is the Apache text and appendix). The carve-out is therefore unresolved from repository files alone. |
| **Purpose** | Plug-and-play image inpainting via decomposed dual-branch diffusion (ECCV 2024, arXiv 2403.06976). ARC Lab Tencent PCG + CUHK. PixelForge role: fallback inpainting backend, **Priority C**, contingent on Phase 4/5 outcomes. |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Hugging Face `TencentARC/BrushEdit/tree/main/brushnetX`; training data `datasets/random123123/BrushData`. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `pyproject.toml`, `setup.py`, `Makefile` |
| **Disk usage** | 78 MB (`.git`: 39 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | Smallest clone of the seven. HEAD dates 2024-12-17. Not scheduled — activated only if Moebius (Phase 4) and PixelHacker (Phase 5) both fail. The unresolved third-party carve-out must be settled before any redistribution or commercial use. |

## 5. ControlNet

| Field | Value |
|---|---|
| **Name** | ControlNet |
| **Remote** | https://github.com/lllyasviel/ControlNet.git |
| **Commit SHA** | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:42:48+0600 |
| **License** | **Apache-2.0** — `LICENSE` (201 lines, verbatim Apache License 2.0). |
| **Purpose** | Adding conditional control to text-to-image diffusion models (arXiv 2302.05543). Copies network blocks into a locked copy and a trainable copy joined by zero-convolutions, so conditioning can be learned without destroying the base model. PixelForge role: structural conditioning and structural preservation during editing (Phase 12). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | Hugging Face `lllyasviel/ControlNet`. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `environment.yaml` (Conda) |
| **Disk usage** | 229 MB (`.git`: 128 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | **Oldest pin of the seven — HEAD dates 2023-09-09.** README states this is ControlNet **1.0** and that a nightly ControlNet 1.1 exists in a separate repository (`lllyasviel/ControlNet-v1-1-nightly`) not yet merged here. Ships a Conda `environment.yaml` from the 2023 CUDA-era ecosystem; expect a substantial dependency audit against Python 3.12/arm64 at Phase 12. Deferred — not MVP. |

## 6. InstructPix2Pix

| Field | Value |
|---|---|
| **Name** | InstructPix2Pix |
| **Remote** | https://github.com/timothybrooks/instruct-pix2pix.git |
| **Commit SHA** | `0dffd1eeb02611c35088462d1df88714ce2b52f4` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:43:50+0600 |
| **License** | **MIT (by text) for the repository's own code** — `LICENSE` (9 lines) is the verbatim MIT permission text, © 2023 Timothy Brooks, Aleksander Holynski, Alexei A. Efros. ⚠️ The file never names "MIT" explicitly; identifier is inferred from verbatim body text, not a stated identifier. ⚠️ **Weights and derived code: REVIEW REQUIRED.** The same file states portions of code and models — *including pretrained checkpoints fine-tuned from released Stable Diffusion checkpoints* — derive from CompVis/stable-diffusion, that *"further restrictions may apply"*, and directs the reader to `stable_diffusion/LICENSE`. That file (present, 14,385 bytes) is **CreativeML Open RAIL-M**, a use-restricted license. |
| **Purpose** | Instruction-based image editing — edits an image from a natural-language instruction. Built on the CompVis Stable Diffusion codebase (arXiv 2211.09800, UC Berkeley). PixelForge role: instruction-based editing (Phase 12). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | `huggingface.co/runwayml/stable-diffusion-v1-5` (`v1-5-pruned.ckpt`), `huggingface.co/stabilityai/sd-vae-ft-mse-original` (`vae-ft-mse-840000-ema-pruned.ckpt`). Scripts present but **not executed**: `scripts/download_checkpoints.sh`, `scripts/download_pretrained_sd.sh`, `scripts/download_data.sh`. |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | None declared |
| **Dependency files** | `environment.yaml` (Conda) |
| **Disk usage** | 36 MB (`.git`: 17 MB) |
| **Working tree** | Clean — 0 modified files |
| **Notes** | **Oldest HEAD of the seven — 2023-01-31.** README states instructions were *"tested on a GPU with >18GB VRAM"* — **at or above this host's entire 18 GB unified memory budget**, which must be shared with the OS. This is an early and concrete signal that InstructPix2Pix may be `CLOUD_GPU`; it is a documented hint, not a measurement. The RAIL-M licensing on weights is the strictest of the seven and constrains permissible use, not merely redistribution. |

## 7. Grounded-Segment-Anything

| Field | Value |
|---|---|
| **Name** | Grounded-Segment-Anything (incl. Grounding DINO) |
| **Remote** | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |
| **Commit SHA** | `126abe633ffe333e16e4a0a4e946bc1003caf757` |
| **Branch** | `main` |
| **Acquisition date** | 2026-08-24T11:44:00+0600 |
| **License** | **Apache-2.0** — `LICENSE` (201 lines, verbatim Apache License 2.0). ⚠️ Scope caveat: this repo composes several independently-licensed upstream models (Grounding DINO, Segment Anything, and optionally LLaVA / RAM). The two declared submodules are **uninitialized**, so their licenses were not inspected and are not covered by this entry. |
| **Purpose** | Combines Grounding DINO (open-set detection) with Segment Anything to detect and segment anything from a text prompt. PixelForge role: text-guided / natural-language selection — "select the dog" — producing candidate boxes that SAM 2 refines into masks (Phase 11). |
| **Environment** | TBD — NOT YET CREATED |
| **Checkpoint source** | `groundingdino_swint_ogc.pth` (GroundingDINO GitHub release v0.1.0-alpha); `sam_vit_h_4b8939.pth` (`dl.fbaipublicfiles.com`). Optional LLaVA weights referenced. **Nothing downloaded.** |
| **Validation status** | NOT YET VALIDATED |
| **Submodules** | ⚠️ **2 declared, both UNINITIALIZED** (per instruction) — `grounded-sam-osx` → `https://github.com/linjing7/grounded-sam-osx.git` @ `6688b036c7856a302f9315bb16864d66fb2cdade`; `VISAM` → `https://github.com/BingfengYan/VISAM` @ `d7c38233882ff9d34d5cbecb8495e175e4dffc8c`. `git submodule status` shows a leading `-` on both, confirming not initialized. Gitlink SHAs above are recorded from the parent tree; the submodule contents are **not on disk**. |
| **Dependency files** | `requirements.txt`, `Makefile`, `Dockerfile` |
| **Disk usage** | 278 MB (`.git`: 160 MB) — **largest of the seven** |
| **Working tree** | Clean — 0 modified files |
| **Notes** | HEAD dates 2024-09-05. Ships a `Dockerfile` and historically requires **CUDA-compiled C++/CUDA extensions** for Grounding DINO — expected to be the hardest device audit of the seven on an MPS-only host. README notes a successor, `Grounded-SAM-2`, which pairs Grounding DINO with SAM 2 directly; **worth evaluating as an alternative at Phase 11** given PixelForge already standardises on SAM 2. Deferred. |

---

## Summary

| # | Repository | Commit SHA | Branch | License | Size | Tree |
|---|---|---|---|---|---|---|
| 1 | SAM 2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | main | Apache-2.0 (+BSD-3-Clause vendored) | 208 MB | clean |
| 2 | PixelHacker | `f5567db2871598aa178fe7a34c520dd478a0b41b` | main | Apache-2.0 | 223 MB | clean |
| 3 | Moebius | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` | main | Apache-2.0 (code **and** weights) | 165 MB | clean |
| 4 | BrushNet | `0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d` | main | Apache-2.0; third-party carve-out **REVIEW REQUIRED** | 78 MB | clean |
| 5 | ControlNet | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` | main | Apache-2.0 | 229 MB | clean |
| 6 | InstructPix2Pix | `0dffd1eeb02611c35088462d1df88714ce2b52f4` | main | MIT by text; weights **CreativeML Open RAIL-M — REVIEW REQUIRED** | 36 MB | clean |
| 7 | Grounded-Segment-Anything | `126abe633ffe333e16e4a0a4e946bc1003caf757` | main | Apache-2.0 (submodules uninspected) | 278 MB | clean |

**7 of 7 acquired. 7 of 7 pinned. 7 of 7 clean. 0 of 7 validated. 0 of 7 classified. 0 weights downloaded. 0 environments created.**

Total upstream disk usage: **1.2 GB**.

### Licensing attention required

| Repository | Issue |
|---|---|
| BrushNet | Declares a third-party exception to Apache-2.0 but does not enumerate the components. Unresolvable from repo files. |
| InstructPix2Pix | Own code MIT *by text only* (never named). Checkpoints derive from Stable Diffusion under **CreativeML Open RAIL-M** — use-restricted, not merely attribution-restricted. Strictest of the seven. |
| Grounded-Segment-Anything | Composite project; two uninitialized submodules have uninspected licenses. |
| SAM 2 | Secondary `LICENSE_cctorch` (BSD-3-Clause) applies to a vendored component. |

Moebius is the only repository giving an unambiguous, permissive position on **weights** as well as code.

## Re-verification

To confirm these pins have not drifted:

```
cd /Users/atik/Projects/PixelForge/research/upstream
for r in sam2 PixelHacker Moebius BrushNet ControlNet instruct-pix2pix Grounded-Segment-Anything; do
  printf "%-28s %s  %s\n" "$r" "$(git -C $r rev-parse HEAD)" "$(git -C $r status --porcelain | wc -l | tr -d ' ') changed"
done
```

Compare against [`LOCKFILE.md`](LOCKFILE.md). Any mismatch means a working tree was altered or updated, which invalidates every result derived from it until re-recorded.

# Upstream Research Repository Registry

**Authoritative provenance record for every third-party research repository used by PixelForge.**

STATUS: **PHASE 1 — REGISTRY CREATED, NOTHING ACQUIRED**

---

## Policy

1. Everything under `research/upstream/` is an **external dependency** and is **READ-ONLY**. PixelForge code must never modify upstream research code; integration happens through adapters in `models/adapters/`.
2. Upstream clones are **not committed** to this repository (see `.gitignore`). Each carries its own `.git` history and its own license. This file is the only tracked artifact in `research/upstream/`.
3. **Commit SHAs are never invented.** A SHA is recorded only after being read from an actual local clone via `git rev-parse HEAD`.
4. **Licenses are never assumed.** A license is recorded only after reading the `LICENSE`/`COPYING` file in the acquired clone. Publicly-remembered licensing is not evidence.
5. Model weights are never committed. `checkpoint source` records where a weight came from, so it can be re-acquired and integrity-checked.
6. A repository is classified `LOCAL_MPS`, `CLOUD_GPU`, `CPU`, or `UNAVAILABLE` only after passing: repository audit → dependency audit → device audit → checkpoint audit → minimal smoke test → performance test → integration test.

Unknown values are recorded verbatim as `TBD — NOT YET ACQUIRED`. No field is filled speculatively.

### Provenance note on the remotes below

The seven remote URLs were recovered from the surviving `.git/config` files of a previous PixelForge working tree (`/Users/atik/atik/venv/Pixelhacker`) whose contents were deleted on 2026-08-24. The URLs are therefore verified as *what was previously cloned*. **No commit SHA survived that deletion**, which is precisely why every `commit` field below is unknown and must be re-established by acquisition in Phase 2.

---

## 1. SAM 2

| Field | Value |
|---|---|
| **name** | SAM 2 (Segment Anything Model 2) |
| **remote** | https://github.com/facebookresearch/sam2.git |
| **purpose** | Promptable segmentation. Intended primary segmentation backend: click-based object selection, mask proposal, and mask refinement. Also the intended refinement stage behind text-guided selection. |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | A previous working tree reportedly executed SAM 2.1 Hiera-Tiny on Apple MPS. That tree was deleted before it could be audited, so the claim is **unverified** and carries no weight here. Phase 3 must re-establish it from scratch as the first reproducibility gate. |

## 2. PixelHacker

| Field | Value |
|---|---|
| **name** | PixelHacker |
| **remote** | https://github.com/hustvl/PixelHacker.git |
| **purpose** | Generative inpainting. Namesake of the project and a candidate inpainting backend (Priority B). |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | A Hugging Face cache entry for `hustvl/PixelHacker` survived the deletion containing a single 40-byte ref pointer, `012fd343158936a265b8a0ee38a791a7a2841f45`. This is a **Hugging Face weights-repo revision, NOT the GitHub source commit** — it must not be recorded in the `commit` field above. No weight files survived. Phase 5 must determine local-MPS vs cloud-GPU viability; treat as likely `CLOUD_GPU` until proven otherwise. |

## 3. Moebius

| Field | Value |
|---|---|
| **name** | Moebius |
| **remote** | https://github.com/hustvl/Moebius.git |
| **purpose** | Generative inpainting. Candidate inpainting backend (**Priority A**) — to be evaluated first for local MPS execution. |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | First inpainting backend to be assessed (Phase 4). Acceptance requires producing a valid image on MPS **without modifying the research algorithm**. |

## 4. BrushNet

| Field | Value |
|---|---|
| **name** | BrushNet |
| **remote** | https://github.com/TencentARC/BrushNet.git |
| **purpose** | Generative inpainting with dual-branch conditioning. Fallback inpainting backend (Priority C) if Moebius and PixelHacker are both unviable locally. |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | Not scheduled before Phase 4 concludes. |

## 5. ControlNet

| Field | Value |
|---|---|
| **name** | ControlNet |
| **remote** | https://github.com/lllyasviel/ControlNet.git |
| **purpose** | Structural conditioning and structural preservation during editing. Intended for advanced editing (Phase 12). |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | Deferred to Phase 12. Not required for MVP. |

## 6. InstructPix2Pix

| Field | Value |
|---|---|
| **name** | InstructPix2Pix |
| **remote** | https://github.com/timothybrooks/instruct-pix2pix.git |
| **purpose** | Instruction-based image editing from natural-language edit instructions. Intended for advanced editing (Phase 12). |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | Deferred to Phase 12. Not required for MVP. |

## 7. Grounded-Segment-Anything

| Field | Value |
|---|---|
| **name** | Grounded-Segment-Anything (incl. Grounding DINO) |
| **remote** | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |
| **purpose** | Open-vocabulary / text-guided object grounding — natural-language selection such as "select the dog". Intended to produce boxes that SAM 2 then refines into masks (Phase 11). |
| **license** | TBD — NOT YET ACQUIRED |
| **commit** | TBD — NOT YET ACQUIRED |
| **date acquired** | TBD — NOT YET ACQUIRED |
| **environment** | TBD — NOT YET ACQUIRED |
| **checkpoint source** | TBD — NOT YET ACQUIRED |
| **validation status** | TBD — NOT YET ACQUIRED |
| **notes** | Contains git submodules and historically carries CUDA-compiled extensions; dependency and device audits are expected to be non-trivial. Deferred to Phase 11. |

---

## Summary

| # | Repository | Acquired | Commit pinned | License verified | Classification |
|---|---|---|---|---|---|
| 1 | SAM 2 | No | No | No | TBD — NOT YET ACQUIRED |
| 2 | PixelHacker | No | No | No | TBD — NOT YET ACQUIRED |
| 3 | Moebius | No | No | No | TBD — NOT YET ACQUIRED |
| 4 | BrushNet | No | No | No | TBD — NOT YET ACQUIRED |
| 5 | ControlNet | No | No | No | TBD — NOT YET ACQUIRED |
| 6 | InstructPix2Pix | No | No | No | TBD — NOT YET ACQUIRED |
| 7 | Grounded-Segment-Anything | No | No | No | TBD — NOT YET ACQUIRED |

**0 of 7 repositories acquired.** No clone, no checkout, no weight download has occurred.

## Acquisition procedure (Phase 2)

For each repository, after cloning into `research/upstream/<name>/`:

```
git -C research/upstream/<name> rev-parse HEAD          # -> commit
git -C research/upstream/<name> log -1 --format=%cI     # -> upstream commit date
ls research/upstream/<name>/LICENSE*                    # -> read, then record license
```

Record the resulting values in this file and commit **this file only**. The clone itself stays untracked.

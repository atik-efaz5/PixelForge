# Research Stack Plan

STATUS: **1 OF 7 VALIDATED — SAM 2 PASSED ITS GATE AND IS CLASSIFIED `LOCAL_MPS`.**

All seven research components have been cloned and pinned to exact commit SHAs (see [`research/upstream/LOCKFILE.md`](upstream/LOCKFILE.md)). **Acquisition is not validation.** Phase 3 ran the SAM 2 gate on real hardware and returned `PASS`: real segmentation inference executed on the Apple M3 Pro GPU via MPS. SAM 2 is therefore `VALIDATED` / `LOCAL_MPS`. The remaining six `Status` fields still read `NOT YET VALIDATED`, and none will change until that component has passed the full validation gate on real hardware.

Provenance (remote, commit SHA, license) is tracked separately and authoritatively in [`research/upstream/REPOSITORIES.md`](upstream/REPOSITORIES.md). This document covers intended role and validation state.

---

## No compatibility is claimed beyond what has been measured

Only SAM 2 has been measured. For the other six components, nothing here asserts that they run on this host, in any configuration.

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

**Purpose:** Generative inpainting. Project namesake and candidate inpainting backend (**Priority B**).

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 5.

**Notes:** Phase 5 must determine whether a local path exists at all. Acceptance is *either* a verified local path *or* a verified cloud-GPU path — both are acceptable outcomes; an unverified assumption is not. A surviving Hugging Face cache entry records a weights-repo revision for `hustvl/PixelHacker`, which is **not** the upstream GitHub commit and is recorded separately in `REPOSITORIES.md`.

## 3. Moebius

**Purpose:** Generative inpainting. Candidate inpainting backend (**Priority A**) — first to be evaluated.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 4.

**Notes:** Evaluated ahead of the other inpainting backends. Acceptance requires producing a valid image **without modifying the research algorithm**. Training-only and CUDA-only dependencies are to be avoided where they are not required for inference.

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

**Purpose:** Instruction-based editing driven by natural-language edit instructions.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 12. Not required for MVP.

**Notes:** Phase 2 recorded two constraints from the repository itself. Its README states the pipeline was *"tested on a GPU with >18GB VRAM"* — at or above this host's **entire** 18 GB unified memory pool, which is also shared with the OS. Its checkpoints derive from Stable Diffusion under **CreativeML Open RAIL-M**, a use-restricted license. Both point toward `CLOUD_GPU`, but neither is a measurement and the classification remains undetermined.

## 7. Grounded-Segment-Anything

**Purpose:** Open-vocabulary text-guided object grounding (including Grounding DINO) — natural-language selection such as "select the dog". Intended to produce candidate boxes that SAM 2 refines into masks.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 11.

**Notes:** Confirmed at Phase 2: two git submodules are declared (`grounded-sam-osx`, `VISAM`) and both remain **uninitialized** by design. Historically ships CUDA-compiled extensions, so the dependency and device audits are expected to be the most involved of the seven. Upstream also documents a successor project pairing Grounding DINO with SAM 2 directly, which is worth evaluating as an alternative at Phase 11 given PixelForge already standardises on SAM 2.

---

## Status summary

| # | Component | Intended role | Priority | Target phase | Status |
|---|---|---|---|---|---|
| 1 | SAM 2 | Segmentation | Core / MVP | 3 | **VALIDATED — `LOCAL_MPS`** |
| 2 | PixelHacker | Inpainting | B | 5 | NOT YET VALIDATED |
| 3 | Moebius | Inpainting | A | 4 | NOT YET VALIDATED |
| 4 | BrushNet | Inpainting | C | contingent | NOT YET VALIDATED |
| 5 | ControlNet | Structural conditioning | Later | 12 | NOT YET VALIDATED |
| 6 | InstructPix2Pix | Instruction editing | Later | 12 | NOT YET VALIDATED |
| 7 | Grounded-Segment-Anything | Text-guided grounding | Later | 11 | NOT YET VALIDATED |

**1 of 7 validated. 7 of 7 acquired and pinned. 1 of 7 classified.**

## MVP dependency

The MVP requires exactly **two** validated components — one segmentation backend and one inpainting backend:

```
UPLOAD → CLICK → SAM 2 SEGMENTATION → EDITABLE MASK
  → ONE VALIDATED INPAINTING BACKEND → RESULT
    → BEFORE / MASK / AFTER → PERFORMANCE METADATA
```

That is SAM 2 (Phase 3) plus the first inpainting backend to pass its gate (Moebius at Phase 4, else PixelHacker at Phase 5, else BrushNet). **SAM 2 is done**, so exactly one inpainting backend now stands between the project and the MVP. The remaining four components are explicitly **not** MVP blockers and must not be started before the MVP path works end to end.

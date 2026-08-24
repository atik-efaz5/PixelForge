# Research Stack Plan

STATUS: **PHASE 2 — ALL 7 ACQUIRED AND PINNED. NO COMPONENT VALIDATED.**

All seven research components have been cloned and pinned to exact commit SHAs (see [`research/upstream/LOCKFILE.md`](upstream/LOCKFILE.md)). **Acquisition is not validation.** None has been installed or executed, no dependency environment exists, and no checkpoint has been downloaded. Every `Status` field reads `NOT YET VALIDATED`, and none will change until that component has passed the full validation gate on real hardware.

Provenance (remote, commit SHA, license) is tracked separately and authoritatively in [`research/upstream/REPOSITORIES.md`](upstream/REPOSITORIES.md). This document covers intended role and validation state.

---

## No compatibility is claimed

Nothing here asserts that any component runs on this host, in any configuration.

- `Execution` is `TBD` because no dependency or device audit has been performed.
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

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 3 — first reproducibility gate for the whole project.

**Notes:** A previous, now-deleted working tree reportedly ran SAM 2.1 Hiera-Tiny on Apple MPS. That tree was destroyed before it could be audited, so the report is **unverified** and is treated as a hypothesis to re-establish, not as prior art. Phase 3 acceptance requires a real inference executing on this machine.

Phase 2 surfaced one favourable piece of evidence: the pinned commit's own subject is an Apple MPS bug fix in `SAM2Base` (upstream #495). That raises the prior that MPS is a supported path upstream. It is **not** a substitute for the Phase 3 measurement.

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
| 1 | SAM 2 | Segmentation | Core / MVP | 3 | NOT YET VALIDATED |
| 2 | PixelHacker | Inpainting | B | 5 | NOT YET VALIDATED |
| 3 | Moebius | Inpainting | A | 4 | NOT YET VALIDATED |
| 4 | BrushNet | Inpainting | C | contingent | NOT YET VALIDATED |
| 5 | ControlNet | Structural conditioning | Later | 12 | NOT YET VALIDATED |
| 6 | InstructPix2Pix | Instruction editing | Later | 12 | NOT YET VALIDATED |
| 7 | Grounded-Segment-Anything | Text-guided grounding | Later | 11 | NOT YET VALIDATED |

**0 of 7 validated. 7 of 7 acquired and pinned. 0 of 7 classified.**

## MVP dependency

The MVP requires exactly **two** validated components — one segmentation backend and one inpainting backend:

```
UPLOAD → CLICK → SAM 2 SEGMENTATION → EDITABLE MASK
  → ONE VALIDATED INPAINTING BACKEND → RESULT
    → BEFORE / MASK / AFTER → PERFORMANCE METADATA
```

That is SAM 2 (Phase 3) plus the first inpainting backend to pass its gate (Moebius at Phase 4, else PixelHacker at Phase 5, else BrushNet). The remaining four components are explicitly **not** MVP blockers and must not be started before the MVP path works end to end.

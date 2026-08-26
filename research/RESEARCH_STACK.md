# Research Stack Plan

STATUS: **PHASE 1 — NO MODELS ACQUIRED OR VALIDATED**

This document lists the seven research components PixelForge intends to evaluate. **Acquisition is not validation.** No `Status` field will change until that component has passed the full validation gate on real hardware.

Provenance (remote, commit SHA, license) will be tracked authoritatively in [`research/upstream/REPOSITORIES.md`](upstream/REPOSITORIES.md) and [`research/upstream/LOCKFILE.md`](upstream/LOCKFILE.md) once repositories are acquired in Phase 2.

---

## No compatibility is claimed

Nothing here asserts that any component runs on this host, in any configuration.

- `Execution` is `TBD` until a dependency and device audit is performed.
- `Local/cloud` is `TBD — NOT YET DETERMINED` because placement is an empirical outcome.
- `Purpose` describes the **intended role** in PixelForge, not observed behaviour.

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

---

## 1. SAM 2

**Purpose:** Promptable segmentation. Intended primary segmentation backend — click-based object selection, mask proposal, and mask refinement.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 3 — first reproducibility gate.

## 2. PixelHacker

**Purpose:** Generative inpainting with structural and semantic consistency. Project namesake and candidate inpainting backend (**Priority B**).

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 5.

## 3. Moebius

**Purpose:** Lightweight generative inpainting. Candidate inpainting backend (**Priority A**) — first to be evaluated.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 4.

## 4. BrushNet

**Purpose:** Generative inpainting with dual-branch conditioning. Fallback inpainting backend (**Priority C**).

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Contingent on Phase 4 and Phase 5 outcomes.

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

## 7. Grounded-Segment-Anything / Grounding DINO

**Purpose:** Open-vocabulary text-guided object grounding — natural-language selection producing candidate boxes that SAM 2 refines into masks.

**Execution:** TBD

**Local/cloud:** TBD — NOT YET DETERMINED

**Status:** **NOT YET VALIDATED**

**Target phase:** Phase 11.

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

**0 of 7 validated. 0 of 7 acquired. 0 of 7 classified.**

## MVP dependency

The MVP requires exactly **two** validated components — one segmentation backend and one inpainting backend:

```
UPLOAD → CLICK → SAM 2 SEGMENTATION → EDITABLE MASK
  → ONE VALIDATED INPAINTING BACKEND → RESULT
    → BEFORE / MASK / AFTER → PERFORMANCE METADATA
```

Neither component is validated yet.

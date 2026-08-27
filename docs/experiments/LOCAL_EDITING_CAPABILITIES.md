# Localized Editing Capabilities

**Phase 12B — Masked object replacement / localized editing audit.**

This document records what PixelForge can honestly offer today for localized editing after object selection, based on source inspection of the Moebius adapter and upstream inference API. It does not claim capabilities the validated backend does not provide.

---

## SUPPORTED

| Capability | Backend | Notes |
|---|---|---|
| **Object selection (click)** | SAM 2 (`LOCAL_MPS`) | Point prompt → bool H×W mask via `POST /segment` |
| **Object selection (text)** | Grounding DINO → SAM 2 | Text finds a box; SAM 2 segments it via `POST /select-by-text`. This text is for **selection only**, not edit instructions. |
| **Mask refinement** | CPU (boolean ops) | Brush, erase, clear, optional dilate/erode in `remove-object` |
| **Localized mask inpainting** | Moebius (`LOCAL_MPS`, `CONDITIONAL`) | Fill or remove the masked region using mask-conditioned diffusion only. Exposed via `POST /inpaint`, `POST /remove-object`, and the UI **Fill selected region** action. |
| **Capability discovery** | API | `GET /capabilities` returns declared edit intents per backend (`pipelines/editing_capabilities.py`). |
| **Pipeline contract** | `ImageEditPipeline.edit_localized()` | Formalizes localized inpaint; rejects non-empty `instruction` with `UnsupportedEditIntentError`. |

### Evidence — Moebius accepts mask only, not user text

1. **`models/adapters/moebius_adapter.py`** — `infer(image, mask, params)`; no prompt or instruction parameter.
2. **`research/upstream/Moebius/removal/v1_2/pipeline.py`** — `RemovalSDXLPipeline_BatchMode.__call__` takes `input_image_list` and `input_mask_list` plus diffusion knobs only.
3. **`research/upstream/Moebius/removal/v1_2/pipeline.py`** — `input_ids` are fixed at init from `range(half_id_num)` / `range(half_id_num, id_num)` for classifier-free guidance; they index `nn.Embedding(20, 3072)`, not CLIP tokens.
4. **`docs/experiments/MOEBIUS_MPS_VALIDATION.md`** — `text_encoder_required: false`; no tokenizer on the inference path.

### User workflow (honest MVP)

```
Select object (click or text-for-selection)
        ↓
   Mask appears
        ↓
 Refine mask (optional)
        ↓
 Fill selected region  →  Moebius inpaints masked pixels
```

---

## NOT SUPPORTED

| Capability | Reason |
|---|---|
| **Semantic object replacement** | Moebius has no natural-language conditioning at inference. Instructions like *"replace it with a red wooden chair"* are ignored by the model API. |
| **Edit instruction field** | Not exposed in UI or API for Moebius. `edit_localized(instruction=...)` raises `UnsupportedEditIntentError`. |
| **`POST /edit` with instruction** | Not added — would mislead clients into sending text the backend cannot use. |
| **Reference-image conditioning** | No adapter or upstream path for per-edit reference images. |
| **Per-user text prompts during generation** | Moebius uses fixed learned embedding indices for CFG, not user-supplied tokens. |

---

## PLANNED

| Capability | Candidate backend | Status |
|---|---|---|
| **Global instruction editing** | InstructPix2Pix | Feasibility documented in `INSTRUCT_PIX2PIX_FEASIBILITY.md`; not implemented |
| **Semantic masked replacement** | InstructPix2Pix or future text-conditioned inpainting | Requires a backend that accepts edit instructions at inference |
| **Cloud inpainting** | PixelHacker | `CLOUD_GPU` boundary only; not wired to localized edit flow |
| **ControlNet / BrushNet** | — | Out of scope for Phase 12B |

---

## Architecture separation (Phase 12B)

| Layer | Responsibility | Phase 12B |
|---|---|---|
| 1. Object selection | SAM 2 click, Grounding DINO + SAM 2 text | Unchanged |
| 2. Mask refinement | Boolean mask ops | Unchanged |
| 3. Edit intent | `EditIntent.LOCALIZED_INPAINT` vs `SEMANTIC_REPLACE` | Only localized inpaint is active |
| 4. Generation backend | Moebius via `inpaint()` / `edit_localized()` | Unchanged inference path |

---

## API summary

| Endpoint | Purpose |
|---|---|
| `GET /capabilities` | List declared edit intents per backend |
| `POST /inpaint` | Image + mask → localized fill (existing) |
| `POST /remove-object` | Click → segment → inpaint (existing) |
| `POST /segment` | Click selection (existing) |
| `POST /select-by-text` | Text **selection** only (existing) |

---

## Related validation

- Moebius MPS: `docs/experiments/MOEBIUS_MPS_VALIDATION.md`
- MVP E2E: `docs/experiments/MVP_END_TO_END_VALIDATION.md`
- InstructPix2Pix feasibility: `docs/experiments/INSTRUCT_PIX2PIX_FEASIBILITY.md`

# InstructPix2Pix Integration

**Phase 13A — global instruction-editing backend boundary (adapter + API + UI).**

This document describes the first PixelForge integration path for InstructPix2Pix. **No inference was executed in this phase.** Runtime validation remains **CONDITIONAL / NOT RUNTIME-VALIDATED**.

---

## Supported capability

| Intent | Backend | Status |
|---|---|---|
| **GLOBAL_INSTRUCTION_EDIT** | `instruct_pix2pix` | Adapter + API + UI wired; cloud endpoint required |
| **MASK_CONDITIONED_EDIT** | — | **Not supported** by InstructPix2Pix upstream |

InstructPix2Pix edits the **full frame** from `image + instruction`. It does not accept a user mask on the upstream inference path (see Phase 12A audit).

---

## Cloud backend requirement

| Property | Value |
|---|---|
| Classification | `CLOUD_GPU` |
| Local MPS | **Not offered** by this adapter |
| Availability | `PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT` or `configs/models.yaml` `endpoint` |
| When unconfigured | `ModelUnavailableError` → HTTP 503 `model_unavailable` |
| When configured but not implemented | `ModelInferenceError` → HTTP 500 `inference_failed` |

The adapter does **not** download checkpoints or run CUDA locally.

---

## Checkpoint requirements (not downloaded)

| File | Path (project-relative) | Notes |
|---|---|---|
| Main weights | `checkpoints/instruct_pix2pix/instruct-pix2pix-00-22000.ckpt` | ~7.7 GB; lives on cloud worker |
| Config | `configs/generate.yaml` (upstream) | Referenced in `configs/models.yaml` |
| Upstream pin | `research/upstream/instruct-pix2pix` @ `0dffd1ee…` | READ-ONLY |

---

## Adapter boundary

| Module | Role |
|---|---|
| `models/adapters/instruct_pix2pix_adapter.py` | `InstructionEditAdapter` implementation |
| `models/adapters/base.py` | `InstructionEditAdapter` contract (distinct from `InpaintingAdapter`) |
| `models/types.py` | `InstructionEditParams`, `InstructionEditResult` |

**Input:** RGB `uint8` H×W×3, non-empty instruction string, optional generation params.

**Output:** RGB `uint8` edited image, latency, backend, model name, metadata.

`infer()` raises until a cloud worker implements remote execution.

---

## Configuration

`configs/models.yaml`:

```yaml
instruct_pix2pix:
  backend: CLOUD_GPU
  endpoint: ""   # or PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT
  checkpoint: checkpoints/instruct_pix2pix/instruct-pix2pix-00-22000.ckpt
  upstream_dir: research/upstream/instruct-pix2pix
```

No secrets are stored in the repository.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /edit-by-instruction` | Image + instruction → edited PNG |
| `GET /capabilities` | Declares `global_instruction_edit` for `instruct_pix2pix` |

Form fields: `image`, `instruction`, optional `num_steps`, `guidance_text`, `guidance_image`, `resolution`.

---

## Pipeline

`ImageEditPipeline.edit_by_instruction(image, instruction, backend="instruct_pix2pix", params=None)`:

1. Validates image and non-empty instruction
2. Rejects `mask` with `UnsupportedEditIntentError`
3. Resolves `instruct_pix2pix` adapter via registry
4. Calls `adapter.infer(image, instruction, params)`

---

## Frontend

Separate **Instruction Edit** panel (global edit). **Localized Fill** (Moebius mask inpaint) remains independent.

When the cloud backend is unavailable, the API error message is shown — no fake output.

---

## Current runtime status

| Check | Result |
|---|---|
| Adapter registered | Yes |
| API endpoint | Yes |
| Unit tests (mocked) | Yes |
| Cloud worker | **Not built** |
| Inference validated | **No** |
| Upstream modified | **No** |

---

## Related documents

- Phase 12A feasibility: `INSTRUCT_PIX2PIX_FEASIBILITY.md`
- Phase 12B localized editing: `LOCAL_EDITING_CAPABILITIES.md`

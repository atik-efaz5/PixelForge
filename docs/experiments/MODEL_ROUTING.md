# Intelligent Model Routing (Phase 18)

Deterministic capability-aware routing for PixelForge operations. The router
**selects** adapters; adapters **perform** inference. No ML ranking, no queues.

## Capabilities

| Capability | Models (priority order) | Backend |
|------------|----------------------|---------|
| `object_selection_point` | sam2 | LOCAL_MPS |
| `object_selection_text` | grounding_dino | CPU |
| `object_selection_box` | sam2 | LOCAL_MPS |
| `localized_inpaint` | moebius, pixelhacker | LOCAL_MPS, CLOUD_GPU |
| `masked_inpaint` | pixelhacker | CLOUD_GPU |
| `global_instruction_edit` | instruct_pix2pix | CLOUD_GPU |

Runtime-validated locally: **sam2**, **grounding_dino**, **moebius**.  
Cloud models are configured but **not** runtime-validated in this MVP.

## Routing rules

1. **No silent operation substitution** — `global_instruction_edit` never becomes `localized_inpaint`.
2. **Explicit backend** — when the client passes `moebius`, `grounding_dino`, etc., the router validates capability + availability only.
3. **Automatic** — aliases: `auto`, `automatic`, or empty → router picks from capability table.
4. **Unavailable** — raises `RoutingError` → pipeline surfaces `ModelUnavailableError` with human-readable reason.
5. **Operation/capability mismatch** — rejected at route time.

## Execution preferences

| Preference | Behavior |
|------------|----------|
| `local_first` | Prefer LOCAL_MPS / CPU before CLOUD_GPU |
| `cloud_first` | Prefer CLOUD_GPU when available |
| `fastest_available` | Prefer runtime-validated local models |
| `quality_first` | Prefer later-listed (typically cloud) candidates when available |

Default: `local_first`.

## Fallback behavior

`RoutingDecision.fallbacks` lists other candidates for the capability with availability flags.  
If the selected backend is unavailable, routing fails — **no silent fallback to a different capability**.

Automatic routing skips unavailable cloud models unless `cloud_first` and the cloud adapter reports available.

## API

### `GET /routing`

Returns capability → model mappings, operation requirements, execution preferences, and live `available` / `status` probes (lightweight — does not load weights).

### Pipeline integration

`ImageEditPipeline` calls `models.router.route()` for:

- `segment` → point selection / SAM2
- `select_by_text` → grounding_dino
- `inpaint` → localized inpaint backend
- `edit_by_instruction` → instruct_pix2pix

Routing metadata is attached under `metadata.routing` on results.

## Frontend

- Inpainting backend selector: **Automatic (router)** or **Moebius (manual)**.
- Manual override preserved for debugging.
- API form defaults remain explicit (`moebius`, `grounding_dino`, `instruct_pix2pix`) for backward compatibility.

## Explicit vs automatic

| Client sends | Router behavior |
|--------------|-----------------|
| `auto` / `automatic` | Select best candidate per preference |
| `moebius` | Must support `localized_inpaint` and be available |
| Wrong model for capability | Clear error, no substitution |

## Limitations

- No PixelHacker / InstructPix2Pix cloud runtime in this repo
- No latency/quality measurement — preferences use static tables only
- Single routing pass per operation (no multi-hop planning)
- SAM2 box step after grounding is fixed to SAM2 (not separately routed)

## Module

- `models/router.py` — types, `route()`, `list_routing_catalog()`
- Tests: `tests/unit/test_model_router.py`

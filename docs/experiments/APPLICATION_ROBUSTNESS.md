# Application Robustness (Phase 15)

This document records hardening work for the PixelForge MVP. It does **not** claim production readiness.

## Validated edge cases

### Image upload (backend + frontend)

| Case | Behavior |
|------|----------|
| Zero-byte upload | HTTP 400 `invalid_input` — empty upload message |
| Corrupted / non-image bytes | HTTP 400 — unrecognized or decode failure |
| Unsupported MIME (`text/plain`) | HTTP 400 when `Content-Type` is set |
| Image smaller than 8×8 | HTTP 400 with dimension in message |
| Image over 16M pixels | HTTP 400 — exceeds max pixel count |
| Grayscale (L) input | Normalized to RGB uint8 |
| RGBA input | Alpha composited via PIL `convert("RGB")` |
| Wide / tall aspect ratios | Accepted when within pixel bounds |
| Client pre-check | `validateImageFile()` rejects empty, wrong type, tiny, oversized before API call |

### Canvas coordinates

- Letterboxed layout via `computeDisplayLayout` / `pointerToImageCoords`
- Clicks outside the rendered image return `null` (no API call)
- Resize and device pixel ratio handled in `EditorCanvas` (CSS size vs canvas backing store)
- Backend validates segment/remove-object points against decoded image bounds

### Mask safety

- Mask must match image H×W (`validate_mask`)
- Empty inpaint region rejected at `/inpaint` before inference
- PNG decode threshold: pixel ≥ 128 → inpaint
- Binary mask PNG roundtrip tested (lossless for bool regions)

### Generation / operation states

Frontend explicit phases (`deriveEditorPhase`):

| Phase | When |
|-------|------|
| `idle` | No image workflow active |
| `uploading` | Client validating / loading upload |
| `segmenting` | SAM2 or Grounding DINO in flight |
| `mask_ready` | Non-empty mask, select tool |
| `refining` | Brush or eraser tool active |
| `generating` | Moebius or instruction edit in flight |
| `result_ready` | Pending inpaint result awaiting accept/discard |
| `error` | Last operation failed |

Duplicate operations blocked while `status !== "idle"` (`busy`).

### Result lifecycle

- Original upload `File` is never mutated; inpaint returns a new blob URL
- Discard calls `EditSessionHistory.releaseResult()` to revoke object URLs
- Session dispose on unmount / new upload revokes accumulated blob URLs
- Accept records `RESULT_ACCEPTED` without leaking prior result URLs

### API validation

- Malformed multipart → FastAPI 422 `validation_error` (no stack trace)
- Invalid coordinates → 400 before segmentation
- Invalid mask / empty mask → 400 `invalid_input`
- Invalid inpaint params (steps, guidance, strength, etc.) → 400
- Unsupported / unavailable backend → 400/503 via existing pipeline handlers
- Unhandled exceptions → 500 `internal_error` generic message (logged server-side)

### Error model

Structured response (backward compatible):

```json
{
  "error": { "code": "invalid_input", "message": "..." },
  "code": "invalid_input",
  "message": "..."
}
```

Frontend `parseErrorMessage` prefers nested `error.message`, then legacy `message`.

### Backend lifecycle

- `GET /health` — static `{"status":"ok"}`; does not load models
- `GET /models` — lightweight registry probe (`available` / `loaded` flags only)
- Isolated Moebius / Grounding DINO subprocesses use `tempfile.TemporaryDirectory`
- Subprocess timeout: `PIXELFORGE_SUBPROCESS_TIMEOUT` (default 600s) → `inference_failed`
- Non-zero exit codes surfaced as `ModelInferenceError` without raw stderr in production responses

### Accessibility (no new libraries)

- `role="alert"` on error banner
- `aria-live="polite"` on status indicators
- `aria-label` on file input, canvas, and primary actions
- `aria-pressed` on tool toggle buttons
- Visible focus via existing button styles

## Resource cleanup

| Resource | Cleanup |
|----------|---------|
| Upload object URLs | Revoked on new upload / unmount |
| Result blob URLs | Revoked on discard / session dispose |
| Mask preview URLs | Replaced on each preview update |
| Subprocess temp dirs | `TemporaryDirectory` context manager |
| Isolated worker PNG I/O | Deleted with temp dir |

## Limitations

- No request rate limiting or auth
- Large images rejected by pixel cap, not progressive downscale
- Instruction edit (InstructPix2Pix) remains API-only / cloud intent — not local inference
- Subprocess timeout does not kill orphaned GPU work inside worker envs
- Frontend coordinate tests mirrored in Python; no browser automation in CI
- Feather preview is visual only; inpaint uses hard bool mask

## Tests added

- `tests/unit/test_robustness.py` — upload, mask, params, error body
- `tests/unit/test_coordinates.py` — letterbox coordinate mapping

Run:

```bash
python -m unittest discover -s tests/unit -p "test_*.py" -v
cd apps/frontend && npm run typecheck && npm run build
```

## Upstream integrity

`research/upstream/*` was not modified in this phase.

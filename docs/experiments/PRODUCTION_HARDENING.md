# Production hardening (Phase 23)

This phase targets **production-quality local deployment**, not multi-tenant
internet-scale deployment.

## Security controls

| Control | Implementation |
|---------|----------------|
| Upload byte limits | `PIXELFORGE_MAX_UPLOAD_BYTES`, chunked reads in `read_upload_bytes()` |
| Decompression bombs | `PIL.Image.MAX_IMAGE_PIXELS` cap before decode |
| Content verification | Magic-byte sniffing for PNG/JPEG/WebP images; PNG-only masks |
| Path traversal | `sanitize_filename()` strips `../` and unsafe characters |
| Temp file safety | `tempfile.TemporaryDirectory` + `_safe_temp_path()` in isolated runner |
| Prompt bounds | `PIXELFORGE_MAX_PROMPT_LENGTH`, `PIXELFORGE_MAX_INSTRUCTION_LENGTH` |
| Error leakage | Generic `internal_error` for unhandled exceptions; no tracebacks in JSON |
| CORS | Explicit origin allowlist via `PIXELFORGE_CORS_ORIGINS`; narrowed headers |
| Worker IPC | Max JSON line size; malformed responses rejected |
| Secrets | No secrets in code; checkpoint paths via env vars only |

## Resource limits

Configured in `apps/backend/settings.py` (environment-variable overrides):

| Setting | Default | Env var |
|---------|---------|---------|
| Max image upload | 25 MB | `PIXELFORGE_MAX_UPLOAD_BYTES` |
| Max mask upload | 10 MB | `PIXELFORGE_MAX_MASK_UPLOAD_BYTES` |
| Max image dimension | 4096 px | `PIXELFORGE_MAX_IMAGE_DIMENSION` |
| Max image pixels | 4096² | `PIXELFORGE_MAX_IMAGE_PIXELS` |
| Max prompt length | 512 | `PIXELFORGE_MAX_PROMPT_LENGTH` |
| Max candidate count | 2 | `PIXELFORGE_MAX_CANDIDATE_COUNT` |
| Concurrent generations | 1 | `PIXELFORGE_MAX_CONCURRENT_GENERATIONS` |
| Worker request timeout | 600 s | `PIXELFORGE_WORKER_REQUEST_TIMEOUT` |
| Subprocess timeout | 600 s | `PIXELFORGE_SUBPROCESS_TIMEOUT` |

## Concurrency policy

- `generation_slot()` semaphore limits concurrent inpaint / remove-object /
  instruction-edit requests (default: 1).
- Second concurrent generation receives HTTP **503** `service_busy`.
- Persistent Moebius worker uses `threading.Lock` for one IPC request at a time.
- In-process MPS adapters enforce a single loaded local model via
  `ModelAdapter._local_resident`.

## Worker lifecycle

1. **Startup** — persistent worker spawned on first isolated Moebius request;
   ready event waited up to `PIXELFORGE_WORKER_STARTUP_TIMEOUT`.
2. **Request** — newline-delimited JSON over stdin/stdout; stderr drained in a
   background thread to prevent pipe stalls.
3. **Timeout** — worker process terminated (then killed) on request timeout.
4. **Crash** — `WorkerCrashedError` → HTTP 503; one-shot subprocess fallback.
5. **Shutdown** — FastAPI lifespan + `atexit` send shutdown command and terminate.

## Configuration

All deployment-specific values load from environment variables through
`get_settings()` in `apps/backend/settings.py`. See table above.

Frontend mirrors upload limits in `apps/frontend/lib/imageUpload.ts`.

## Logging

`RequestContextMiddleware` assigns `X-Request-ID` and logs:

```
request_complete id=… method=… path=… status=… latency_ms=…
generation_start / generation_end operation=… request_id=…
```

Logs do **not** include uploaded image bytes, secrets, or full prompts.

## Data lifecycle

- Upload bytes are held in memory only for the request duration.
- Subprocess temp directories use `TemporaryDirectory` context managers.
- Frontend revokes candidate blob URLs on accept/discard (`editSession.ts`).
- No persistent user storage is introduced.

## Limitations

- Single-machine concurrency only; no distributed queue.
- Generation slot does not queue — excess requests are rejected.
- Persistent worker still ignores custom `InpaintParams` in the MVP bridge.
- MPS determinism and OOM recovery are best-effort.
- This is not hardened for public multi-tenant internet exposure without
  additional reverse-proxy auth, TLS termination, and rate limiting.

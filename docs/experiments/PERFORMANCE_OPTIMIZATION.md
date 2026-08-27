# Local Performance Optimization (Phase 17)

Measurement and application-layer lifecycle optimization for the PixelForge MVP.
Does **not** change model weights, algorithms, or upstream research code.

## Baseline (Phase 16)

Authoritative baseline: [`MVP_BENCHMARK.md`](MVP_BENCHMARK.md) / [`evaluation/reports/mvp_benchmark.json`](../../evaluation/reports/mvp_benchmark.json) at commit `dee3564965e38d5e354855767ceca3a037a1889b`.

| Path | Total | Dominant stage |
|------|-------|----------------|
| Click → SAM2 → Moebius | ~38.7 s | Moebius ~37.8 s |
| Text → Grounding DINO → SAM2 → Moebius | ~52.6 s | Moebius ~42.7 s |

Component steady-state (warm, in-env):

| Component | Warm latency | Load (cold) |
|-----------|--------------|-------------|
| SAM2 point | ~0.18 s | ~1.5 s |
| Grounding DINO | ~2.3 s | ~4.7 s |
| Moebius infer | ~27–31 s | ~5.7 s |

**Bottleneck:** Moebius diffusion inference dominates. Application overhead (subprocess + reload) was a secondary cost on the FastAPI path.

## Bottleneck analysis (code inspection)

### Why Moebius uses an isolated subprocess

FastAPI runs in `pixelforge-sam2-v2`. Moebius requires `pixelforge-moebius` (different PyTorch/diffusers stack). The service catches `ModelLoadError` and calls `inpaint_via_isolated_env()` in `apps/backend/services.py`.

### Previous lifecycle (per request)

```
FastAPI (sam2 env)
  → subprocess.run(pixelforge-moebius/python, isolated_inpaint_worker.py)
      → adapter.load()      # ~5–7 s
      → adapter.infer()     # ~27–31 s
      → adapter.unload()
      → process exit
```

**Answers from code:**

1. Isolated subprocess: conda environment isolation (not architecture choice).
2. Subprocess started: **yes, every request** (one-shot worker).
3. Model loaded: **yes, every request** in one-shot worker (`scripts/isolated_inpaint_worker.py` loads then unloads).
4. Persistent worker: **feasible** — same env, model stays loaded between requests.
5. FastAPI IPC: **yes** — newline-delimited JSON on stdin/stdout; no research imports in FastAPI.
6. SAM2: loads per request in pipeline but stays hot within a session when not unloaded; not optimized in this phase.
7. One-model-at-a-time: **preserved** — SAM2 in API process, Moebius in separate worker process (no co-residency in one Python interpreter).
8. Worker keeps model loaded across requests: **yes** after this change.
9. Expected savings: eliminate per-request load + subprocess spawn on warm requests.

## Architecture change

### Before

```
FastAPI → subprocess.run → isolated_inpaint_worker.py → load → infer → unload → exit
```

### After (default)

```
FastAPI → LineJsonWorkerClient → moebius_persistent_worker.py (long-lived)
           model loaded once → infer → infer → … → shutdown on app exit
```

Fallback: one-shot subprocess if persistent worker fails or `PIXELFORGE_MOEBIUS_PERSISTENT_WORKER=0`.

### New files

| File | Role |
|------|------|
| `apps/backend/persistent_worker.py` | Generic JSON-line worker client (start, request, timeout, shutdown) |
| `scripts/moebius_persistent_worker.py` | Long-lived Moebius worker in `pixelforge-moebius` |

### Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIXELFORGE_MOEBIUS_PERSISTENT_WORKER` | `1` | Use persistent worker |
| `PIXELFORGE_WORKER_STARTUP_TIMEOUT` | `180` | Worker ready timeout (seconds) |
| `PIXELFORGE_WORKER_REQUEST_TIMEOUT` | `600` | Per-request timeout (seconds) |

## Measured before / after (case1_isolated_disc)

Real comparison on Apple M3 Pro, same image/mask as Phase 16 case 1 (`outputs/mvp_benchmark/` artifacts). **3 warm persistent requests** after one one-shot baseline.

| Mode | Request | Wall time | Notes |
|------|---------|-----------|-------|
| **Before** one-shot subprocess | 1 | **40,781 ms** | load + infer + exit |
| **After** persistent worker | 1 (cold worker) | 37,843 ms | worker spawn + load + infer |
| **After** persistent worker | 2 (warm) | **30,698 ms** | no reload |
| **After** persistent worker | 3 (warm) | **30,267 ms** | no reload |

**Warm savings vs one-shot:** ~10.5 s per request (~26% of Moebius wall time), from eliminating per-request model load and subprocess startup.

Inference-only (adapter `infer_ms` on warm requests): ~30.2 s — unchanged from baseline (algorithm not modified).

## Memory

| Mode | Behavior |
|------|----------|
| Before | Moebius RSS released after each subprocess exit |
| After | Moebius worker holds ~1.5 GB RSS + MPS allocations for session lifetime |
| SAM2 + Moebius | Still **separate processes** — no dual LOCAL_MPS load in one interpreter |

## Output equivalence

Same deterministic test image, mask, and default params (`num_steps=20`):

| Run | Output SHA-256 |
|-----|----------------|
| One-shot | `4d827c6aaea83de45a7491a5264b5502f8e9b1dcbdd918b4ba144940c06fd901` |
| Persistent ×3 | **identical** |

`output_hashes_match: true` in `outputs/mvp_benchmark/lifecycle_comparison.json`.

> Phase 16 benchmark hash (`5a90baba…`) differs from this session — MPS diffusion can vary between process lifetimes. Within a single comparison session all paths matched.

## Worker reliability

- Serial requests (thread lock) — one inference at a time
- Request timeout → worker terminated
- Crash → fallback to one-shot subprocess on next request
- FastAPI lifespan + `atexit` → graceful `shutdown` command
- Upstream load prints redirected to stderr during worker boot (protocol-safe stdout)
- Temp PNG dirs still cleaned per request (`TemporaryDirectory`)

## Limitations

- **Does not reduce diffusion steps** — ~30 s infer remains the floor
- First request in a session still pays worker spawn + model load
- Worker memory resident until API shutdown
- Grounding DINO still one-shot subprocess (not optimized; ~2.3 s vs ~30 s Moebius)
- SAM2 not changed in this phase
- Not production multi-tenant hardened (single local worker)

## Tests

- `tests/unit/test_persistent_worker.py` — lifecycle, routing, timeout, no Moebius import in FastAPI
- Full suite: `python -m unittest discover -s tests/unit -p "test_*.py" -v`

## Upstream integrity

`research/upstream/*` unchanged in this phase.

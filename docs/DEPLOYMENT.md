# PixelForge Deployment Guide

**Phase 25 — local / self-hosted deployment packaging**

This guide packages PixelForge for reproducible deployment on a single machine.
It does **not** cover internet-scale, multi-tenant, or cloud-GPU production.

> **Scope:** robust local deployment on macOS with Apple Silicon (MPS).
> Linux and CUDA hosts are not validated in this release.

---

## Table of contents

1. [Supported host assumptions](#1-supported-host-assumptions)
2. [Required vs optional vs not available](#2-required-vs-optional-vs-not-available)
3. [Prerequisites](#3-prerequisites)
4. [Repository layout](#4-repository-layout)
5. [Model environments](#5-model-environments)
6. [Checkpoints](#6-checkpoints)
7. [Configuration](#7-configuration)
8. [Startup sequence](#8-startup-sequence)
9. [Health checks](#9-health-checks)
10. [Persistent Moebius worker](#10-persistent-moebius-worker)
11. [Shutdown](#11-shutdown)
12. [Diagnostics](#12-diagnostics)
13. [Troubleshooting](#13-troubleshooting)
14. [Recovery](#14-recovery)
15. [Docker](#15-docker)

---

## 1. Supported host assumptions

| Assumption | Validated value |
|------------|-----------------|
| OS | macOS 14+ (Darwin arm64) |
| CPU/GPU | Apple Silicon (M-series) with Metal / MPS |
| RAM | ≥ 18 GB unified memory recommended |
| Storage | ≥ 50 GB free for repos + checkpoints |
| CUDA | **Not available** — not required for local MVP |
| Network | Required once for checkpoint acquisition; not required at runtime |

Validated development host: **Apple M3 Pro, 18 GB RAM, macOS 15.x**.

---

## 2. Required vs optional vs not available

### REQUIRED (click + inpaint MVP)

| Component | Role |
|-----------|------|
| Git clone of PixelForge | Application source |
| Conda env `pixelforge-sam2-v2` | FastAPI backend + SAM 2 in-process |
| Conda env `pixelforge-moebius` | Moebius isolated inpaint worker |
| `.e2e_deps/` | Vendored FastAPI/uvicorn stack for backend |
| Node.js + npm | Frontend build/dev server |
| SAM 2 checkpoint | Segmentation |
| Moebius student + VAE | Inpainting |
| `npm install` in `apps/frontend` | Frontend dependencies |

### OPTIONAL

| Component | Role |
|-----------|------|
| Conda env `pixelforge-grounding-dino` | Text-based object selection |
| Grounding DINO checkpoint | Required if text selection is used |
| Persistent Moebius worker | Enabled by default; disable with `PIXELFORGE_MOEBIUS_PERSISTENT_WORKER=0` |
| Production frontend build | `npm run build && npm run start` instead of dev server |

### NOT AVAILABLE LOCALLY

| Component | Classification | Notes |
|-----------|----------------|-------|
| PixelHacker | `CLOUD_GPU` | No local MPS path; needs remote endpoint |
| InstructPix2Pix | `CLOUD_GPU` | No local runtime validated |
| ControlNet / BrushNet | Research only | Upstream cloned; no adapter |
| Authentication / multi-tenant | Not implemented | Local single-user assumed |
| Kubernetes / Redis / PostgreSQL | Not in scope | Single-process local deployment |

---

## 3. Prerequisites

### Python

- **Backend interpreter:** Python **3.11.x** in conda env `pixelforge-sam2-v2`
- **Moebius worker:** Python **3.11.x** in conda env `pixelforge-moebius`
- **Grounding DINO worker (optional):** Python **3.11.x** in conda env `pixelforge-grounding-dino`

Environment creation is documented in [`docs/ENVIRONMENT_PLAN.md`](ENVIRONMENT_PLAN.md).
The startup scripts **do not** create conda environments automatically.

### Node / npm

- **Node.js** ≥ 18 (validated with v24.x)
- **npm** ≥ 9

Install frontend dependencies once:

```bash
cd apps/frontend
npm install
cp .env.example .env.local   # optional; defaults to http://127.0.0.1:8000
```

### Backend Python dependencies (`.e2e_deps`)

The backend runs FastAPI/uvicorn from a vendored tree at the repository root:

```bash
# From repository root — only if .e2e_deps/ is missing
/opt/anaconda3/envs/pixelforge-sam2-v2/bin/pip install \
  -t .e2e_deps \
  fastapi uvicorn python-multipart httpx pyyaml
```

The `.e2e_deps/` directory is gitignored. It must exist before starting the backend.

---

## 4. Repository layout

```
PixelForge/
├── apps/
│   ├── backend/          # FastAPI application
│   └── frontend/       # Next.js UI
├── checkpoints/        # Model weights (NOT in git)
├── configs/
│   ├── models.yaml     # Model paths and metadata
│   └── deployment.env.example
├── docs/
│   └── DEPLOYMENT.md   # This file
├── models/             # Adapter layer
├── research/upstream/  # Pinned research repos (do not modify)
├── scripts/
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   ├── stop_local.sh
│   ├── check_environment.py
│   └── check_models.py
└── .e2e_deps/          # Vendored backend deps (NOT in git)
```

Set `PIXELFORGE_ROOT` if the repository is not at the default inferred path.

---

## 5. Model environments

PixelForge uses **separate conda environments per model**. Do not merge them.

| Environment | Python path (default) | Used for |
|-------------|----------------------|----------|
| `pixelforge-sam2-v2` | `$CONDA_PREFIX/bin/python` or `/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python` | Backend API, SAM 2 |
| `pixelforge-moebius` | `/opt/anaconda3/envs/pixelforge-moebius/bin/python` | Moebius inpaint worker |
| `pixelforge-grounding-dino` | `/opt/anaconda3/envs/pixelforge-grounding-dino/bin/python` | Grounding DINO worker |

Override interpreter paths with environment variables:

| Variable | Purpose |
|----------|---------|
| `PIXELFORGE_BACKEND_PYTHON` | Backend + SAM 2 interpreter |
| `PIXELFORGE_MOEBIUS_PYTHON` | Moebius worker interpreter |
| `PIXELFORGE_GROUNDING_DINO_PYTHON` | Grounding DINO worker interpreter |

If conda is installed somewhere other than `/opt/anaconda3`, set these explicitly.

---

## 6. Checkpoints

Checkpoints live under `checkpoints/` relative to the repository root.
They are **never committed to git** and are **never downloaded automatically**.

### Expected locations

| Model | Default path | Env override |
|-------|--------------|--------------|
| SAM 2.1 Hiera-Tiny | `checkpoints/sam2/sam2.1_hiera_tiny.pt` | `PIXELFORGE_SAM2_CHECKPOINT` |
| Moebius student | `checkpoints/moebius/ft_places2/diffusion_pytorch_model.bin` | `PIXELFORGE_MOEBIUS_CHECKPOINT` |
| Moebius VAE | `checkpoints/moebius/vae/` (directory) | `PIXELFORGE_MOEBIUS_VAE` |
| Grounding DINO | `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` | `PIXELFORGE_GROUNDING_DINO_CHECKPOINT` |

### Verify checkpoints (no model load)

```bash
cd /path/to/PixelForge
/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python scripts/check_models.py
/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python scripts/check_models.py --sha
```

Optional `--sha` computes SHA-256 (slow for large files). Known validation
anchors from prior phases are shown when present.

Acquisition instructions: [`docs/ENVIRONMENT_PLAN.md`](ENVIRONMENT_PLAN.md) and
model-specific validation docs under `docs/experiments/`.

---

## 7. Configuration

All backend settings load from environment variables via
[`apps/backend/settings.py`](../apps/backend/settings.py).
Copy [`configs/deployment.env.example`](../configs/deployment.env.example)
and export values before starting services.

### Core networking

| Variable | Default | Description |
|----------|---------|-------------|
| `PIXELFORGE_API_HOST` | `127.0.0.1` | Backend bind address |
| `PIXELFORGE_API_PORT` | `8000` | Backend port |
| `NEXT_PUBLIC_API_BASE_URL` | `http://127.0.0.1:8000` | Frontend → backend URL |
| `PIXELFORGE_FRONTEND_PORT` | `3000` | Frontend dev server port |
| `PIXELFORGE_CORS_ORIGINS` | localhost:3000,5173 | Comma-separated browser origins |

### Upload and image limits

| Variable | Default |
|----------|---------|
| `PIXELFORGE_MAX_UPLOAD_BYTES` | 26214400 (25 MiB) |
| `PIXELFORGE_MAX_MASK_UPLOAD_BYTES` | 10485760 (10 MiB) |
| `PIXELFORGE_MAX_IMAGE_DIMENSION` | 4096 |
| `PIXELFORGE_MAX_IMAGE_PIXELS` | 16777216 |
| `PIXELFORGE_MAX_PROMPT_LENGTH` | 512 |

### Concurrency and workers

| Variable | Default | Description |
|----------|---------|-------------|
| `PIXELFORGE_MAX_CONCURRENT_GENERATIONS` | `1` | Reject second inpaint with 503 |
| `PIXELFORGE_MOEBIUS_PERSISTENT_WORKER` | `true` | Long-lived Moebius worker |
| `PIXELFORGE_WORKER_STARTUP_TIMEOUT` | `180` | Worker ready timeout (seconds) |
| `PIXELFORGE_WORKER_REQUEST_TIMEOUT` | `600` | Per-request worker timeout |
| `PIXELFORGE_SUBPROCESS_TIMEOUT` | `600` | One-shot subprocess timeout |

### Cloud endpoints (optional, not locally validated)

| Variable | Model |
|----------|-------|
| `PIXELFORGE_PIXELHACKER_ENDPOINT` | PixelHacker remote worker |
| `PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT` | InstructPix2Pix remote worker |

### Path overrides

| Variable | Purpose |
|----------|---------|
| `PIXELFORGE_ROOT` | Repository root if auto-detection fails |
| `PIXELFORGE_SAM2_CHECKPOINT` | SAM 2 weights |
| `PIXELFORGE_MOEBIUS_CHECKPOINT` | Moebius student weights |
| `PIXELFORGE_MOEBIUS_VAE` | Moebius VAE directory |
| `PIXELFORGE_GROUNDING_DINO_CHECKPOINT` | Grounding DINO weights |

**Do not hardcode machine-specific absolute paths in source files.**
Use environment variables or relative paths in `configs/models.yaml`.

---

## 8. Startup sequence

### Pre-flight checks

```bash
cd /path/to/PixelForge

# Environment diagnostic (does not install anything)
python scripts/check_environment.py

# Checkpoint diagnostic (does not load weights)
python scripts/check_models.py
```

### Terminal 1 — Backend

```bash
cd /path/to/PixelForge
./scripts/start_backend.sh
```

Equivalent manual command:

```bash
cd /path/to/PixelForge
export PYTHONPATH=".e2e_deps:$PWD"
export PIXELFORGE_API_HOST=127.0.0.1
export PIXELFORGE_API_PORT=8000
/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python \
  -m uvicorn apps.backend.main:app \
  --host 127.0.0.1 --port 8000
```

Wait for: `Application startup complete.`

### Terminal 2 — Frontend

```bash
cd /path/to/PixelForge
./scripts/start_frontend.sh
```

Equivalent manual command:

```bash
cd /path/to/PixelForge/apps/frontend
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev -- --port 3000
```

### Open the application

Browser: **http://127.0.0.1:3000**

The frontend talks to the backend at `NEXT_PUBLIC_API_BASE_URL`.

---

## 9. Health checks

These endpoints do **not** load model weights.

```bash
# Basic health (version, limits, request ID)
curl -sS -D - http://127.0.0.1:8000/health -o /dev/null

# Model availability (reads config + checkpoint presence)
curl -sS http://127.0.0.1:8000/models | python3 -m json.tool

# Routing table
curl -sS http://127.0.0.1:8000/routing | python3 -m json.tool

# Capability summary
curl -sS http://127.0.0.1:8000/capabilities | python3 -m json.tool
```

Expected `/health` response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "max_upload_bytes": 26214400,
  "max_concurrent_generations": 1
}
```

Response includes header `X-Request-ID`.

Frontend health: `curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/`
should return `200`.

---

## 10. Persistent Moebius worker

When `PIXELFORGE_MOEBIUS_PERSISTENT_WORKER=true` (default):

1. First Moebius inpaint request spawns `scripts/moebius_persistent_worker.py`
   in the `pixelforge-moebius` environment.
2. The worker loads Moebius weights once and keeps them resident.
3. Subsequent inpaints reuse the warm worker (~26% latency reduction vs one-shot).
4. IPC is newline-delimited JSON over stdin/stdout.
5. On worker crash or timeout, the backend falls back to one-shot subprocess mode.

Disable persistent worker (one-shot subprocess every request):

```bash
export PIXELFORGE_MOEBIUS_PERSISTENT_WORKER=0
```

Worker process appears as:

```
pixelforge-moebius/bin/python .../scripts/moebius_persistent_worker.py
```

---

## 11. Shutdown

### Graceful (recommended)

1. **Frontend:** press `Ctrl+C` in Terminal 2.
2. **Backend:** press `Ctrl+C` in Terminal 1.
   - FastAPI lifespan sends shutdown to persistent workers.
   - `atexit` handlers terminate worker subprocesses.

### Script-assisted stop

```bash
cd /path/to/PixelForge
./scripts/stop_local.sh
```

This sends SIGTERM to processes listening on the configured backend/frontend
ports and to any orphan `moebius_persistent_worker` processes.

### Verify clean shutdown

```bash
pgrep -fl "uvicorn apps.backend.main|moebius_persistent_worker|next dev" || echo "clean"
```

No orphan worker processes should remain.

---

## 12. Diagnostics

| Script | Purpose |
|--------|---------|
| `scripts/check_environment.py` | Python, Node, OS, MPS, env paths, key packages |
| `scripts/check_models.py` | Checkpoint presence, size, optional SHA-256 |
| `scripts/operational_readiness_check.py` | Full operational validation (Phase 24) |

Run from repository root. None of these install packages or download weights.

Example:

```bash
python scripts/check_environment.py --json
python scripts/check_models.py --sha
```

---

## 13. Troubleshooting

### Backend fails: `.e2e_deps/` missing

```
ERROR: .e2e_deps/ not found
```

Install vendored deps (see §3) or restore from backup.

### Backend fails: Python env not found

```
ERROR: Backend Python not found (pixelforge-sam2-v2)
```

Create the conda environment per `docs/ENVIRONMENT_PLAN.md`, or set
`PIXELFORGE_BACKEND_PYTHON=/path/to/python`.

### SAM 2 unavailable / 503 on segment

- Verify checkpoint: `python scripts/check_models.py`
- Confirm SAM 2 env: `python scripts/check_environment.py`

### Moebius inpaint fails / slow first request

- First request loads model (~30 s on M3 Pro); subsequent requests are faster
  with persistent worker.
- Check `pixelforge-moebius` env and Moebius checkpoint + VAE directory.
- Inspect backend stderr for `in-process Moebius load failed; using isolated env`
  (expected — Moebius always runs in isolated env on validated setup).

### Text selection fails

- Requires `pixelforge-grounding-dino` env and Grounding DINO checkpoint.
- Grounding runs on **CPU** by design.

### CORS errors in browser

Add your frontend origin to `PIXELFORGE_CORS_ORIGINS`:

```bash
export PIXELFORGE_CORS_ORIGINS="http://127.0.0.1:3000,http://localhost:3000"
```

Restart backend after changing CORS.

### Port already in use

```bash
lsof -i :8000
lsof -i :3000
./scripts/stop_local.sh
```

Or use alternate ports:

```bash
export PIXELFORGE_API_PORT=8001
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001
export PIXELFORGE_FRONTEND_PORT=3001
```

### Stale server returning old `/health` format

Kill orphan processes from prior sessions before starting:

```bash
./scripts/stop_local.sh
```

### Orphan Moebius worker after crash

```bash
pgrep -fl moebius_persistent_worker
kill <pid>
# or
./scripts/stop_local.sh
```

---

## 14. Recovery

### Git / source recovery

```bash
git clone https://github.com/atik-efaz5/PixelForge.git
cd PixelForge
git checkout main   # or specific release tag/commit
```

Verify upstream integrity:

```bash
for d in research/upstream/*/; do git -C "$d" status --short; done
# Should produce no modified tracked files
```

### Checkpoint recovery

Checkpoints are not in git. Re-acquire from documented sources in
`docs/ENVIRONMENT_PLAN.md` and verify with:

```bash
python scripts/check_models.py --sha
```

Compare SHA-256 against anchors in:

- `docs/experiments/SAM2_MPS_VALIDATION.md`
- `docs/experiments/MOEBIUS_MPS_VALIDATION.md`
- `docs/experiments/GROUNDING_DINO_VALIDATION.md`

### Environment recovery

Recreate conda environments per `docs/ENVIRONMENT_PLAN.md`.
Reinstall frontend deps: `cd apps/frontend && npm install`.
Recreate `.e2e_deps/` per §3.

---

## 15. Docker

**Docker is intentionally not provided** for the current Apple Silicon + MPS
deployment path.

Reasons:

1. **MPS requires Metal on the host.** Standard Linux Docker containers cannot
   access Apple MPS. A Docker image would imply CUDA/Linux inference, which is
   a different deployment target not validated in this project.
2. **Multi-environment architecture.** PixelForge uses three separate conda envs
   with incompatible dependencies. Containerizing this without misleading
   users about MPS support adds complexity without improving reproducibility on
   the validated platform.
3. **Local deployment goal.** The validated workflow is two terminal windows on
   macOS — startup scripts and this guide are sufficient.

If Linux/CUDA deployment is needed in the future, it requires a separate
validation phase and dedicated container spec. Do not assume the current
Docker-less packaging is a gap for the M3 Pro local target.

---

## Quick reference

```bash
# Pre-flight
python scripts/check_environment.py
python scripts/check_models.py

# Start (two terminals)
./scripts/start_backend.sh      # Terminal 1
./scripts/start_frontend.sh     # Terminal 2

# Verify
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models

# Stop
./scripts/stop_local.sh
```

For operational validation results, see
[`docs/experiments/PRODUCTION_READINESS.md`](experiments/PRODUCTION_READINESS.md).

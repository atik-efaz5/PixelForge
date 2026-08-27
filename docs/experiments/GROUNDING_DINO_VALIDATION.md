# Grounding DINO Validation

**Date:** 2026-08-27  
**Phase:** 11B — Real grounding inference validation  
**Repository:** `/Users/atik/Projects/PixelForge`  
**Commit:** `279c25a95c35029c5ab7fbc756f3ccb573dd1e7d`  
**Verdict:** **PASS**

---

## Summary

Real open-vocabulary grounding and SAM 2 box segmentation were validated end-to-end on Apple Silicon:

```
text prompt ("circle")
  → Grounding DINO (CPU, isolated env)
  → bounding box
  → SAM 2 box prompt (LOCAL_MPS, in-process)
  → bool H×W mask
```

The FastAPI `/select-by-text` endpoint and the Next.js **Find Object** UI both produced a non-empty mask on the same deterministic fixture image. Moebius inpainting was **not** run in this phase.

---

## Environment

| Role | Environment | Python | Architecture |
|------|-------------|--------|--------------|
| Grounding DINO | `pixelforge-grounding-dino` | **3.11.15** | **arm64** (Apple Silicon) |
| SAM 2 + FastAPI | `pixelforge-sam2-v2` + `.e2e_deps` | 3.11.15 | arm64 |
| Frontend | Node dev server | Node v24.19.0 | arm64 |

**Host:** macOS 15.7.9, Apple M-series (validated in prior phases).

### Pre-install baseline (reference envs)

| Package | `pixelforge-sam2-v2` | `pixelforge-moebius` |
|---------|----------------------|----------------------|
| Python | 3.11.15 | 3.11.15 |
| torch | 2.13.0 | 2.13.0 |

### `pixelforge-grounding-dino` packages installed (minimum CPU path)

| Package | Version |
|---------|---------|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| transformers | **4.38.2** (required; 5.x breaks `BertModelWarper`) |
| timm | 1.0.28 |
| numpy | 2.4.6 |
| pillow | 12.3.0 |
| opencv-python | 5.0.0.93 |
| pycocotools | 2.0.11 |
| supervision | 0.30.1 |
| addict, yapf | per upstream `requirements.txt` |
| groundingdino | 0.1.0 editable from vendored upstream tree |

**Notes:**

- No CUDA packages installed.
- Upstream custom C++ ops are **not** built on macOS; Grounding DINO runs with the upstream CPU fallback (`Failed to load custom C++ ops. Running on CPU mode Only!`).
- `pixelforge-sam2-v2` and `pixelforge-moebius` were **not modified**.

### Environment creation (record)

```bash
conda create -n pixelforge-grounding-dino python=3.11 -y
/opt/anaconda3/envs/pixelforge-grounding-dino/bin/pip install \
  torch torchvision transformers==4.38.2 timm addict yapf numpy \
  opencv-python supervision pycocotools pillow
cd research/upstream/Grounded-Segment-Anything/GroundingDINO
/opt/anaconda3/envs/pixelforge-grounding-dino/bin/pip install -e .
```

---

## Checkpoint

| Property | Value |
|----------|-------|
| File | `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` |
| Source | [GroundingDINO v0.1.0-alpha release](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) |
| Size | **661 MiB** (693,997,677 bytes) |
| SHA-256 | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| Config | `research/upstream/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` |

Checkpoint is git-ignored per project policy.

---

## Test image and prompt

| Property | Value |
|----------|-------|
| Path | `tests/fixtures/mvp_e2e_input.png` |
| Size | 512×512 RGB |
| Content | Deterministic synthetic scene — orange disc on gradient background |
| Prompt | **`circle`** |
| Expected object | Orange disc centered ~(256, 256); orange pixel bbox ≈ (193, 193)–(319, 319) |

---

## Device

| Stage | Device | Classification |
|-------|--------|----------------|
| Grounding DINO | **CPU** (`device: cpu`) | CPU (confirmed) |
| SAM 2 box segmentation | **MPS** (`device: mps` in segmentation metadata) | LOCAL_MPS |

Grounding runs in `pixelforge-grounding-dino` via `scripts/isolated_grounding_worker.py` because `pixelforge-sam2-v2` cannot load Grounding DINO in-process (missing `groundingdino` / incompatible `transformers`).

---

## Grounding result (real inference)

Measured in `pixelforge-grounding-dino` (in-process adapter, cold load):

| Metric | Value |
|--------|-------|
| Model load time | **5406 ms** |
| Grounding inference time | **3144 ms** (adapter metadata) |
| Detections returned | **1** |
| Selected label | **circle** |
| Confidence | **0.941** |
| Box XYXY (pixels) | **[183.89, 183.16, 328.91, 329.09]** |

Isolated subprocess wall time (includes fresh load per worker invocation): **~8.3–9.3 s**.

Thresholds: `box_threshold=0.3`, `text_threshold=0.25`.

---

## SAM 2 handoff

Full `ImageEditPipeline.select_by_text()` via `ImageEditingService` (grounding isolated + SAM 2 in-process):

| Check | Result |
|-------|--------|
| Box in source-image pixel space | **PASS** — coordinates within 512×512 bounds, aligned with orange disc |
| Mask shape | **(512, 512)** |
| Mask dtype | **bool** |
| Mask non-empty | **PASS** — area **16,210** px (~6.2%) |
| Segmentation method | **box** |
| Segmentation model | SAM 2.1 Hiera-Tiny |
| Segmentation confidence | **0.995** |
| Box center inside mask | **PASS** |
| Mask coverage inside grounding box | **75.5%** of box pixels True |
| SAM 2 pipeline stage latency | **~2991 ms** (includes encoder warm load) |

---

## API validation

**Endpoint:** `POST /select-by-text`  
**Backend:** `pixelforge-sam2-v2` on `127.0.0.1:8000`

| Check | Result |
|-------|--------|
| HTTP status | **200** |
| Response body | PNG mask |
| `content-type` | `image/png` |
| `x-pf-prompt` | `circle` |
| `x-pf-detection-count` | `1` |
| `x-pf-selected-label` | `circle` |
| `x-pf-method` | `box` |
| `x-pf-detections` | JSON array with one detection (label, confidence, box_xyxy) |
| Total request time | **13.4 s** (cold grounding subprocess + SAM 2) |

Moebius was **not** called.

---

## Frontend validation

**URL:** `http://127.0.0.1:3000/`  
**Automation:** Playwright (Chromium headless) against running Next.js dev server.

| Step | Result |
|------|--------|
| Upload `mvp_e2e_input.png` | **PASS** |
| Enter prompt `circle` | **PASS** |
| Click **Find Object** | **PASS** |
| Status **Finding object…** shown | **PASS** |
| Mask overlay on canvas | **PASS** (teal overlay on disc) |
| Mask preview in Result panel | **PASS** |
| Auto-switch to Brush tool | **PASS** |
| Brush click on canvas | **PASS** |
| Find Object round-trip time | **~12.0 s** |

Screenshot saved locally (not committed): `outputs/grounding_validation/frontend_after_find_object.png`.

---

## Measured timings (single-run reference)

| Stage | ms |
|-------|-----|
| Grounding DINO model load (CPU) | 5406 |
| Grounding DINO inference (CPU) | 3144 |
| Isolated grounding subprocess wall | 8318–9332 |
| SAM 2 box segmentation stage | 2991 |
| Full `select_by_text` service call | 16047 |
| API `/select-by-text` total | 13408 |
| Frontend Find Object UI | 11962 |

These are **one-shot cold-start figures** on this host. Grounding reloads per isolated worker invocation; not production benchmarks.

---

## Unit tests

```bash
python -m unittest discover -s tests/unit -p "test_*.py" -v
```

**Result:** **62/62 PASS** (no code changes required).

---

## Limitations

1. **CPU-only grounding** — ~3 s inference + ~5 s load per cold worker on this machine; not interactive-fast without caching.
2. **No custom CUDA/C++ ops** on macOS — upstream CPU fallback only; behavior matches Phase 11 audit.
3. **`transformers` pin required** — `transformers==4.38.2`; newer 5.x breaks `BertModelWarper.get_head_mask`.
4. **Isolated subprocess per request** — MVP bridge reloads Grounding DINO each time; wall latency dominates.
5. **Synthetic test image** — prompt `circle` on a drawn disc; real-photo prompts not validated here.
6. **Single detection** — fixture produced one box; multi-detection UI not exercised in this run.

---

## Upstream integrity

`research/upstream/*` — **unchanged** (no diff at validation time).

---

## Verdict rationale

**PASS** — Real Grounding DINO SwinT OGC inference on CPU produced a correct box for prompt `circle`, and real SAM 2 box segmentation produced a non-empty bool H×W mask in source-image space. The FastAPI endpoint and browser UI both exercised the same path successfully.

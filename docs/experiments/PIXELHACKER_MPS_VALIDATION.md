# PixelHacker MPS Validation

**Phase 5 — PixelHacker local MPS reproducibility gate.**

Question this document answers: *can the pinned PixelHacker implementation run real inpainting on this machine using Apple MPS without changing the research algorithm?*

Date of validation: **2026-08-31**
Harness: [`tests/smoke/test_pixelhacker_mps.py`](../../tests/smoke/test_pixelhacker_mps.py)
Machine-readable record: `outputs/pixelhacker_mps_validation/result.json` (git-ignored)
Source classification companion: [`PIXELHACKER_FEASIBILITY.md`](PIXELHACKER_FEASIBILITY.md)

> **Answer: no.** Verdict **`FAIL`** for `LOCAL_MPS`. Primary placement **`CLOUD_GPU`** (runtime not validated).
>
> The host has a working Metal device and PyTorch MPS, but PixelHacker imports `flash-linear-attention` (`fla`) GLA operators at module load, `fla` is not installable on this Apple Silicon host, and the upstream entry point selects `cuda:0` or `cpu` only. No inference was executed. No UNet weights were downloaded. Upstream remained READ-ONLY.

---

## Repository

| Property | Value |
|---|---|
| Path | `research/upstream/PixelHacker` |
| Remote | `https://github.com/hustvl/PixelHacker.git` |
| Treated as | **READ-ONLY** — 0 changed files before and after |
| Installed into environment | **No** (`sys.path` insertion only) |
| `upstream_unmodified` | **`true`** |

## Commit

```
f5567db2871598aa178fe7a34c520dd478a0b41b
```

Verified twice (`observed_commit`, `observed_commit_second_pass`). Matches `research/upstream/LOCKFILE.md`.

## Environment

| Property | Value |
|---|---|
| Interpreter | `/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python` |
| Python | 3.11.15 |
| `pixelforge-pixelhacker` | **not created** (by design) |
| torch | 2.13.0 |
| MPS built / available | **true / true** |
| CUDA available | **false** |
| Metal device creatable | **true** |

The gate used an existing torch+MPS environment to measure host capability. PixelHacker dependencies (`diffusers`, `einops`, `fla`, …) were **not** installed — import failure is the expected blocking outcome.

## Device audit (upstream source)

| Property | Value |
|---|---|
| Entry point | `infer_pixelhacker.py:31` |
| Device line | `device = "cuda:0" if torch.cuda.is_available() else "cpu"` |
| MPS references in `*.py` | **0** |
| Official devices | `cuda:0`, `cpu` |

## Dependency audit

| Package | Pin (`requirements.txt`) | Gate result |
|---|---|---|
| `flash-linear-attention` | 0.3.2 | **`fla` not installed** — `ModuleNotFoundError` |
| `triton` | 3.2.0 | not installable on macOS arm64 (Phase 4 measurement) |
| `flash-attn` | 2.5.8 | listed; not on inference import graph |
| `torch` | 2.3.0 | upstream pin; gate used host torch 2.13.0 for MPS probe only |

GLA import lines in `gla_model/gla.py` (verbatim):

```python
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.models.gla.modeling_gla import GLAMLP, GLABlock, GatedLinearAttention, GLAConfig
from fla.modules import FusedCrossEntropyLoss, RMSNorm, ShortConvolution, FusedRMSNormSwishGate
```

## Import audit (measured)

| Stage | Result | Error |
|---|---|---|
| `fla.ops.gla` | **blocked** | `ModuleNotFoundError: No module named 'fla'` |
| `gla_model/gla.py` | **blocked** | `ModuleNotFoundError: No module named 'einops'` (before `fla` is reached) |
| `gla_model.PixelHacker.PixelHacker` | **blocked** | `ModuleNotFoundError: No module named 'diffusers'` |

Even with PixelHacker dependencies installed, `fla`/Triton remain the architectural blocker on Apple Silicon. Moebius-style teacher import isolation **cannot** be reused: GLA **is** the PixelHacker UNet self-attention.

## Checkpoints

| Artifact | Present | Note |
|---|---|---|
| `weight/ft_places2/diffusion_pytorch_model.bin` | **no** | not downloaded this phase |
| `vae/diffusion_pytorch_model.bin` | **no** | not downloaded this phase |

Weights are not required to record `LOCAL_MPS: FAIL` — import audit fails first.

## Mandated checks

| Check | Result |
|---|---|
| `1_repository_pin_verified` | **PASS** |
| `2_upstream_unmodified` | **PASS** |
| `3_mps_host_available` | **PASS** |
| `4_upstream_has_no_mps_path` | **PASS** |
| `5_fla_not_installable` | **PASS** (negative gate — fla blocked) |
| `6_gla_import_blocked` | **PASS** |
| `7_model_import_blocked` | **PASS** |
| `8_no_inference_executed` | **PASS** (expected) |

## Verdict

| Field | Value |
|---|---|
| **Verdict** | **`FAIL`** |
| **Classification** | **`LOCAL_MPS`** |
| **Primary placement** | **`CLOUD_GPU`** |
| **Failure class** | **A — dependency incompatibility** |
| **Failure stage** | **import** |
| **Cloud runtime validated** | **no** |

This is a **recorded negative result**, not an inconclusive run. The host supports MPS; the pinned PixelHacker architecture does not.

## What this phase did not do

- Download PixelHacker UNet weights
- Create `pixelforge-pixelhacker`
- Deploy a cloud GPU worker
- Run CUDA inference
- Modify `research/upstream/PixelHacker`

## Phase 6 boundary

`PixelHackerAdapter` declares `CLOUD_GPU` capability and endpoint configuration only. Local MVP inpainting remains **Moebius** (`LOCAL_MPS` / `CONDITIONAL`). See `models/adapters/pixelhacker_adapter.py`.

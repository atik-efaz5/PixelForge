# PixelForge Evaluation System

**Phase 14 — measurement and reproducibility layer (no model inference in this phase).**

This document describes the evaluation infrastructure under `evaluation/`. It does **not** rerun SAM 2, Grounding DINO, or Moebius inference. The layer is ready to accept future real experiment outputs.

---

## Principle

**No reference-dependent quality metric is reported without a valid reference.**

If ground truth is missing, reference-dependent metrics return `value: null` with an explicit `kind` and `note` — never fabricated scores and never silent `NaN`.

---

## Available metrics

### Mask metrics (`evaluation/metrics/mask_metrics.py`)

| Metric | Kind | Requires reference | Meaning |
|---|---|---|---|
| `mask_area_ratio` | COMPUTED | No | Fraction of True pixels |
| `iou` | REFERENCE_DEPENDENT | Yes (GT mask) | Intersection / union |
| `precision` | REFERENCE_DEPENDENT | Yes | TP / (TP + FP) |
| `recall` | REFERENCE_DEPENDENT | Yes | TP / (TP + FN) |
| `f1` | REFERENCE_DEPENDENT | Yes | 2PR / (P + R) |
| `boundary_overlap` | REFERENCE_DEPENDENT | Yes | IoU on morphological boundary bands |
| `boundary_difference_ratio` | REFERENCE_DEPENDENT | Yes | Symmetric boundary XOR / union |

Zero denominators → `value: null`, `note` explains why.

### Image metrics (`evaluation/metrics/image_metrics.py`)

| Metric | Kind | Status |
|---|---|---|
| `mse` | COMPUTED | Implemented (uint8 RGB) |
| `psnr` | REFERENCE_DEPENDENT | Implemented: `10 log10(MAX² / MSE)` |
| `ssim` | UNAVAILABLE | scikit-image not a project dependency |
| `lpips` | UNAVAILABLE | LPIPS/torch not installed for evaluation |

### Preservation metrics (`evaluation/metrics/preservation.py`)

| Metric | Kind | Meaning |
|---|---|---|
| `outside_mask_mae` | COMPUTED | Normalized MAE outside edit mask |
| `outside_mask_preservation` | COMPUTED | `1 - outside_mask_mae` |

These measure **pixel preservation outside the mask**, not human-perceived edit quality or semantic correctness.

---

## When metrics are valid

| Question | Valid metrics |
|---|---|
| How accurate is selection/mask? | IoU, precision, recall, F1 — **only with GT mask** |
| How much area is selected? | `mask_area_ratio` — always (given mask) |
| How well was outside region preserved? | `outside_mask_preservation` — original + edited + mask |
| Image fidelity vs reference? | PSNR — reference image required |
| Perceptual quality? | **Not reported** (SSIM/LPIPS unavailable) |

---

## Reproducibility record

`evaluation/reproducibility.py` → `build_reproducibility_record()` captures:

- Python version, OS, architecture
- Model, commit, backend, device, environment
- Generation parameters, seed
- SHA-256 hashes for image / mask / output bytes

Secrets are never recorded.

---

## Experiment runner

`evaluation/runner.py` → `ExperimentRunner`:

1. Accepts `ExperimentConfig` + optional `ExperimentInputs`
2. Runs externally supplied stage functions
3. Records per-stage latency (`MEASURED`)
4. Collects metrics when references exist
5. Emits JSON via `write_report()`

Not a full benchmarking framework — minimal orchestration only.

---

## Report format

Example schema: `evaluation/reports/schema.example.json`

Each metric entry includes:

```json
{
  "name": "iou",
  "value": 0.82,
  "kind": "REFERENCE_DEPENDENT",
  "unit": "ratio",
  "note": null
}
```

Kinds: `MEASURED`, `COMPUTED`, `REFERENCE_DEPENDENT`, `UNAVAILABLE`, `UNDEFINED`.

Unavailable metrics (e.g. SSIM) use `value: null`, `kind: UNAVAILABLE` — **not zero**.

---

## Limitations

- No datasets downloaded; unit tests use synthetic fixtures only
- No database or server-side persistence
- No new model integrations
- SSIM/LPIPS deferred until dependencies are explicitly adopted
- Preservation metrics are pixel-level, not perceptual
- Array hashes are layout-dependent raw-byte SHA-256 (documented in reproducibility record)

---

## Related validation docs

Existing runtime gates (not re-run here):

- `docs/experiments/SAM2_MPS_VALIDATION.md`
- `docs/experiments/GROUNDING_DINO_VALIDATION.md`
- `docs/experiments/MOEBIUS_MPS_VALIDATION.md`
- `docs/experiments/MVP_END_TO_END_VALIDATION.md`

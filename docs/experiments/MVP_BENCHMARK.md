# MVP Local Pipeline Benchmark (Phase 16)

**Status:** COMPLETE  
**Timestamp (UTC):** 2026-08-27T01:07:51.802769+00:00  
**PixelForge commit:** `74112c2da79e284bbbd977092107cd3bc6b2cb23`  

> Measurement-only benchmark. Does not claim production readiness.

## Methodology

- Three synthetic 512×512 scenes generated from closed-form math (no external dataset, no RNG).
- Real inference in isolated conda environments (`pixelforge-sam2-v2`, `pixelforge-grounding-dino`, `pixelforge-moebius`).
- Three warm repetitions for steady-state latency (SAM2 point, Moebius inpaint).
- Metrics: `mask_area_ratio` (COMPUTED), IoU/F1 only where exact reference mask exists (case 1).

## Hardware

- Platform: macOS-15.7.9-arm64-arm-64bit-Mach-O
- Processor: Apple M3 Pro
- Cores: 11 physical / 11 logical

## Model commits

- **sam2:** `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- **moebius:** `b88d462bacb9af6e7128a3b4cc4a07418bedfd61`
- **grounding_dino:** `126abe633ffe333e16e4a0a4e946bc1003caf757`

## Test cases

### case1_isolated_disc
- Simple isolated orange disc on gradient background
- Point: `[320, 256]` | Text: "orange circle"
- Image SHA-256: `7053c08ca2915d7c4652648394791918660cc1df72949f575ccd92c88a1113e9`
- Reference mask: yes

### case2_adjacent_objects
- Red square touching blue square — tests boundary ambiguity
- Point: `[160, 240]` | Text: "red square"
- Image SHA-256: `f274a56177886f2277318e448fb8ac93905b460d26924538457d240951cb3d31`
- Reference mask: no

### case3_textured_background
- Green ellipse on busy sinusoidal background
- Point: `[256, 256]` | Text: "green ellipse"
- Image SHA-256: `10c8d61c9cb6f79718edaa5c8a9cb436519aa63c624347b6a0b104ad878fdf57`
- Reference mask: no


## Benchmark 1 — SAM2 point segmentation

- Model load: **1545.8 ms**
- Peak RSS: 650.59 MiB | MPS driver: 5.7 MiB
- **case1_isolated_disc**: cold 1807.3 ms, warm best 182.7 ms, area 0.0585, confidence 0.9799
- **case2_adjacent_objects**: cold 184.5 ms, warm best 185.2 ms, area 0.0244, confidence 0.9688
- **case3_textured_background**: cold 185.2 ms, warm best 183.9 ms, area 0.0592, confidence 0.9864

## Benchmark 2 — Grounding DINO

- Model load: **4659.1 ms**
- **case1_isolated_disc**: 2412.5 ms, 1 detection(s), selected conf 0.963491
- **case2_adjacent_objects**: 2323.7 ms, 1 detection(s), selected conf 0.802531
- **case3_textured_background**: 2317.3 ms, 1 detection(s), selected conf 0.918812

## Benchmark 3 — SAM2 box (after grounding)

- **case1_isolated_disc**: 497.3 ms, area 0.0585, box `[249, 185, 391, 327]`
- **case2_adjacent_objects**: 199.3 ms, area 0.0487, box `[118, 198, 281, 281]`
- **case3_textured_background**: 188.2 ms, area 0.0593, box `[165, 199, 347, 312]`

## Benchmark 4 — Moebius inpainting

- Model load: **5692.6 ms**
- **case1_isolated_disc**: cold 26836.1 ms, warm best 26922.9 ms, output [512, 512, 3]
- **case2_adjacent_objects**: cold 29372.3 ms, warm best 29829.2 ms, output [512, 512, 3]
- **case3_textured_background**: cold 31255.1 ms, warm best 30837.3 ms, output [512, 512, 3]

## Benchmark 5 — End-to-end

### text_to_moebius
- Total: **52627.5 ms**
  - grounding_dino: 7285.7 ms (cpu)
  - sam2_box: 499.6 ms (mps)
  - moebius: 42678.3 ms (mps)
### click_to_moebius
- Total: **38716.2 ms**
  - sam2_point: 357.7 ms (mps)
  - moebius: 37803.2 ms (mps)

## Limitations

- Measurement only — no optimization applied.
- Cross-env orchestration adds subprocess overhead to end-to-end timings.
- Grounding DINO quality metrics omitted (no ground-truth boxes).
- Moebius marked CONDITIONAL in prior validation; benchmark records actual behavior.
- Generated PNG artifacts are not committed (outputs/mvp_benchmark/).

## Machine-readable report

See [`evaluation/reports/mvp_benchmark.json`](../../evaluation/reports/mvp_benchmark.json).

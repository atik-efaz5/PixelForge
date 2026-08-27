# Real-Image Quality Evaluation (Phase 20)

**Status:** COMPLETE  
**Timestamp (UTC):** 2026-08-27T03:50:08.687369+00:00  
**PixelForge commit:** `efefa0d3b4493fa4b8299104bd2679fbda515b61`  

> Evaluation on deterministic local scenes. Does not claim photographic realism.

## Hardware & environment

- Platform: macOS-15.7.9-arm64-arm-64bit-Mach-O
- Processor: Apple M3 Pro
- Unified memory: 19327352832 bytes

## Test set

7 cases at 512×512 — categories: person, vehicle, furniture, outdoor_scene, textured_background, adjacent_objects, cluttered_scene

| Case | Category | SHA-256 (image) | Reference GT |
|------|----------|-----------------|--------------|
| `real01_person` | person | `7499b40f1978d48a…` | yes |
| `real02_vehicle` | vehicle | `9dd8802fb94aae61…` | yes |
| `real03_furniture` | furniture | `96b7ccc02787be13…` | yes |
| `real04_outdoor` | outdoor_scene | `ad18b7fecbffecc4…` | yes |
| `real05_textured` | textured_background | `1226475e9973b674…` | yes |
| `real06_adjacent` | adjacent_objects | `608fe7896feb8d1c…` | yes |
| `real07_cluttered` | cluttered_scene | `268cb39f3269a0cf…` | yes |

## Selection results

### click_sam2 — MEASURED
- **real01_person**: latency recorded; IoU=0.41996781060173405; mask_area_ratio=0.030857
- **real02_vehicle**: latency recorded; IoU=0.2776209677419355; mask_area_ratio=0.015759
- **real03_furniture**: latency recorded; IoU=0.9964047252182845; mask_area_ratio=0.037018
- … and 4 more (see JSON)

### text_grounding — MEASURED
- **real01_person**: latency recorded; IoU=n/a; mask_area_ratio=n/a
- **real02_vehicle**: latency recorded; IoU=n/a; mask_area_ratio=n/a
- **real03_furniture**: latency recorded; IoU=n/a; mask_area_ratio=n/a
- … and 4 more (see JSON)

### text_grounding_sam2_box — MEASURED
- **real01_person**: latency recorded; IoU=0.7647058823529411; mask_area_ratio=0.056187
- **real02_vehicle**: latency recorded; IoU=0.7337719003826274; mask_area_ratio=0.041763
- **real03_furniture**: latency recorded; IoU=0.9983564458140729; mask_area_ratio=0.03709
- … and 4 more (see JSON)

## Inpainting (Moebius)

Status: **MEASURED**
- `real01_person`: cold=31332.112 ms, warm best=30786.428 ms, output `fba6fd0810c56b3a…`
- `real02_vehicle`: cold=31011.825 ms, warm best=31298.694 ms, output `e48386975c1fe576…`
- `real06_adjacent`: cold=30351.177 ms, warm best=35039.737 ms, output `50b1dbbb5677d73f…`

## Preservation

- `real01_person`: outside_mask_preservation=0.9999590845634141 (pixel-level, not perceptual)
- `real02_vehicle`: outside_mask_preservation=0.9999703921287444 (pixel-level, not perceptual)
- `real06_adjacent`: outside_mask_preservation=0.999974280600058 (pixel-level, not perceptual)

## End-to-end workflows

- **text_to_moebius** (`real03_furniture`): total 55720.613 ms — stages: [{'stage': 'grounding_dino', 'latency_ms': 10069.778, 'device': 'cpu'}, {'stage': 'sam2_box', 'latency_ms': 1089.251, 'device': 'mps'}, {'stage': 'moebius', 'latency_ms': 41476.501, 'device': 'mps'}]
- **click_to_moebius** (`real06_adjacent`): total 31684.245 ms — stages: [{'stage': 'sam2_point', 'latency_ms': 565.66, 'device': 'mps'}, {'stage': 'moebius', 'latency_ms': 30345.07, 'device': 'mps'}]

## Qualitative observations

Subjective dimensions are listed in JSON without numeric scores. Inspect PNG artifacts under `outputs/real_image_eval/`.

## Limitations

- Scenes are deterministic synthetic images, not photographs from a public dataset.
- Reference masks are constructed analytically — metrics measure agreement with designed targets.
- Subjective quality (semantic plausibility, visible artifacts) is recorded but not numerically scored.
- Moebius inpainting runs on a subset (3 cases) to limit inference cost.
- Cross-env subprocess orchestration adds overhead to end-to-end timings.
- Does not supersede or overwrite Phase 16 mvp_benchmark.json.
- Generated PNG artifacts are not committed (outputs/real_image_eval/).

## Machine-readable report

See [`evaluation/reports/real_image_quality.json`](../../evaluation/reports/real_image_quality.json).

Phase 16 baseline: [`evaluation/reports/mvp_benchmark.json`](../../evaluation/reports/mvp_benchmark.json) (unchanged).

# Smart Selection Quality Layer (Phase 21)

Deterministic heuristic ranking for SAM 2 and Grounding DINO → SAM 2 mask
candidates. **Not** a learned quality model — scores are relative rankings only.

## Selection strategies

| Mode | User input | Pipeline path |
|------|------------|---------------|
| **Smart** (default) | Click or text prompt | Ranked multimask / multi-detection |
| **Point** | Click only | Legacy `POST /segment` (single mask) |
| **Text** | Text prompt only | Legacy `POST /select-by-text` |

`select_smart()` never silently remaps an explicit mode. Smart routing uses
available input: text prompt → text path; otherwise point coordinates.

## Heuristic signals

| Signal | Purpose |
|--------|---------|
| `area_sweet_spot` | Prefer object-sized masks (~0.8%–40% of image) |
| `point_contains` | Reject masks that miss the click (point path) |
| `box_iou` / `box_containment` | Prefer masks aligned with grounding box |
| `component_sanity` | Penalize fragmented masks (largest CC ratio) |
| `boundary_smoothness` | Compactness proxy for boundary regularity |
| `sam_confidence` | SAM2 predicted score when available |

**Hard rejections:** empty mask, too small (`< 0.05%`), too large (`> 75%`),
point outside mask.

## Ranking behavior

1. Generate candidates:
   - Point: SAM2 `multimask_output=True` (up to 3 masks)
   - Text: one SAM2 box mask per grounding detection (all ranked)
2. Score each candidate with weighted signal average.
3. Drop rejected candidates.
4. Pick highest score; ties broken by `candidate_id` (deterministic).

## Confidence tiers

| Tier | Threshold | Meaning |
|------|-----------|---------|
| HIGH | score ≥ 0.72 | Strong heuristic agreement |
| MEDIUM | score ≥ 0.45 | Acceptable but review mask |
| LOW | below 0.45 | Weak heuristic match |

Tiers are **not** calibrated probabilities.

## API

`POST /select-smart` — form fields:

- `selection_mode`: `smart` | `point` | `text`
- `x`, `y` (optional)
- `prompt` (optional)
- `detection_index` (optional, text narrowing)

Returns PNG mask + metadata including `confidence_tier`, `ranking`, `method`.

## Limitations

- Heuristics tuned for ~512×512 MVP scenes; not validated on photographs
- SAM2 multimask only on point path; box path uses single mask per detection
- No learned reranker; difficult silhouettes may still score LOW
- Phase 20 benchmark unchanged (`real_image_quality.json`)

## Modules

- `pipelines/selection_quality.py` — scoring and ranking
- `pipelines/orchestration/image_edit_pipeline.py` — `select_smart()`
- `models/adapters/sam2_adapter.py` — `segment_point_candidates()`
- Tests: `tests/unit/test_selection_quality.py`, `tests/unit/test_smart_selection.py`

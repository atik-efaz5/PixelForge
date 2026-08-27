# Candidate generation and ranking (Phase 22)

PixelForge can generate one or two inpainting candidates, rank them with
deterministic heuristics, and let the user choose the preferred result.

**The candidate score is a deterministic heuristic and is not a calibrated
measure of human preference.**

## Candidate workflow

1. User draws or selects a mask and optionally enables **Generate 2 candidates**.
2. `POST /inpaint` receives `candidate_count` (`1` default, `2` when requested).
3. For `candidate_count=1`, the API returns a single PNG with header metadata
   (unchanged from prior phases).
4. For `candidate_count=2`, the API returns `multipart/form-data` with:
   - a JSON `metadata` part (ranking, scores, hashes, seeds)
   - one PNG part per candidate (`candidate_1`, `candidate_2`)
5. The pipeline loads Moebius once and runs sequential inferences with derived
   seeds (no duplicate model processes).
6. The UI shows **Option 1 / Option 2** previews when two candidates are
   returned. The user previews, selects, then accepts or discards.
7. On accept, history records `RESULT_ACCEPTED` with `selectedCandidateId`.
   Discarded candidate blob URLs are revoked.

## Seed strategy

- Base seed: request `seed` form field, or `0` when omitted.
- Derived seeds: `base_seed + index * 10007` for `index` in `0 .. count-1`.
- Moebius adapter sets `torch`, `random`, and `numpy` RNG from each derived
  seed immediately before inference. Global random state is not relied on.
- Seeds are recorded in candidate metadata for reproducibility.

## Ranking signals

Transparent deterministic signals in `evaluation/candidate_ranking.py`:

| Signal | Meaning |
|--------|---------|
| `output_valid` | uint8 H×W×3 image passes validation |
| `finite_pixels` | no NaN/Inf values |
| `correct_dimensions` | output matches input spatial size |
| `preservation_outside_mask` | `1 - outside_mask_mae` when defined |
| `mask_change_magnitude` | inside-mask change in a moderate band |
| `artifact_heuristic` | saturation / luminance variance inside mask |

Unavailable metrics are marked `available: false` with a note and are omitted
from the weighted score (not silently converted to zero).

## Score interpretation

- Output field: **candidate score** (0–1 heuristic aggregate).
- Higher score means better alignment with the deterministic signals above.
- Ranking tie-break: higher score first, then lexicographic `candidate_id`.
- Rejected candidates (invalid output) are listed separately and never selected.

## API changes

`POST /inpaint` optional fields:

- `candidate_count`: `1` (default) or `2`
- `seed`: optional non-negative 31-bit integer

Single-candidate responses remain `image/png` with `x-pf-*` headers.
Two-candidate responses are `multipart/form-data` with JSON + PNG parts only
(no base64 blobs in JSON).

## Limitations

- At most two candidates; no large candidate sets.
- Ranking is heuristic only — not human preference calibration.
- Isolated Moebius fallback supports single-candidate paths only; two-candidate
  generation requires in-process Moebius.
- Seed reproducibility depends on backend determinism; MPS may have minor
  floating-point variance across runs.
- No persistent storage of candidate images; client holds blob URLs until accept
  or discard.

## Reproducibility

Evaluation fragments via `InpaintCandidateRanking.to_eval_record()` include:

- `candidate_count`
- `candidate_hashes`
- `scores`
- `ranking` (ordered ids)
- `selected_candidate_id`

## Memory considerations

- Sequential generation reuses one loaded Moebius adapter.
- Frontend revokes discarded candidate blob URLs on accept/discard.
- Session history does not retain more than two candidate URLs per inpaint step.

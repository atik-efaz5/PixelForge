# Upstream Repository Lockfile

**Machine-verifiable pins for all upstream research repositories.**

Generated: **2026-08-27** (Phase 1 — registry scaffolding only; no clones on disk)

Location: `/Users/atik/Projects/PixelForge/research/upstream/`

**Commit SHAs are the authority.** Branch names and tags are mutable and are recorded for context only once acquired.

Human-readable context (purpose, license analysis, checkpoint sources, notes): [`REPOSITORIES.md`](REPOSITORIES.md).

---

## Pins

| Repository | Directory | Commit SHA | Branch | Acquired at | Working tree |
|---|---|---|---|---|---|
| SAM 2 | `sam2` | **NOT YET ACQUIRED** | — | — | — |
| PixelHacker | `PixelHacker` | **NOT YET ACQUIRED** | — | — | — |
| Moebius | `Moebius` | **NOT YET ACQUIRED** | — | — | — |
| BrushNet | `BrushNet` | **NOT YET ACQUIRED** | — | — | — |
| ControlNet | `ControlNet` | **NOT YET ACQUIRED** | — | — | — |
| InstructPix2Pix | `instruct-pix2pix` | **NOT YET ACQUIRED** | — | — | — |
| Grounded-Segment-Anything | `Grounded-Segment-Anything` | **NOT YET ACQUIRED** | — | — | — |

`Acquired at` will record the creation time of each clone's `.git` directory on this host once Phase 2 completes.

---

## Remotes (planned)

| Repository | Remote |
|---|---|
| SAM 2 | https://github.com/facebookresearch/sam2.git |
| PixelHacker | https://github.com/hustvl/PixelHacker.git |
| Moebius | https://github.com/hustvl/Moebius.git |
| BrushNet | https://github.com/TencentARC/BrushNet.git |
| ControlNet | https://github.com/lllyasviel/ControlNet.git |
| InstructPix2Pix | https://github.com/timothybrooks/instruct-pix2pix.git |
| Grounded-Segment-Anything | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |

---

## Weight / checkpoint state

**No model weights exist.** No repository has been cloned. No download script has been executed.

---

## Rules governing this file

1. This file is **derived**, never hand-edited with invented SHAs. Regenerate it from the clones in Phase 2.
2. A SHA is written only after `git rev-parse HEAD` returned it from a real local clone, verified twice.
3. Tags and branch names are never authoritative.
4. Any change to a pin is a **provenance event**: update this file and [`REPOSITORIES.md`](REPOSITORIES.md) in the same commit, and re-run validation for the affected model.
5. Upstream clones themselves are never committed to this repository — only these registry files are.

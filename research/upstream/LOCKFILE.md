# Upstream Repository Lockfile

**Machine-verifiable pins for all upstream research repositories.**

Generated: **2026-08-24** (Phase 2 — research repository acquisition + provenance)
Location: `/Users/atik/Projects/PixelForge/research/upstream/`

Every SHA in this file was obtained by running `git rev-parse HEAD` inside the actual local clone and was **verified twice**, with both passes producing an identical value. No SHA was inferred from a tag, a release name, a web API, or memory.

**Commit SHAs are the authority.** Branch names and tags are mutable and are recorded for context only. If a branch has moved, the SHA — not the branch — defines what was audited.

Human-readable context (purpose, license analysis, checkpoint sources, notes): [`REPOSITORIES.md`](REPOSITORIES.md).

---

## Pins

| Repository | Directory | Commit SHA | Branch | Acquired at | Working tree |
|---|---|---|---|---|---|
| SAM 2 | `sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `main` | 2026-08-24T11:38:50+0600 | clean |
| PixelHacker | `PixelHacker` | `f5567db2871598aa178fe7a34c520dd478a0b41b` | `main` | 2026-08-24T11:39:47+0600 | clean |
| Moebius | `Moebius` | `b88d462bacb9af6e7128a3b4cc4a07418bedfd61` | `main` | 2026-08-24T11:41:27+0600 | clean |
| BrushNet | `BrushNet` | `0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d` | `main` | 2026-08-24T11:42:19+0600 | clean |
| ControlNet | `ControlNet` | `ed85cd1e25a5ed592f7d8178495b4483de0331bf` | `main` | 2026-08-24T11:42:48+0600 | clean |
| InstructPix2Pix | `instruct-pix2pix` | `0dffd1eeb02611c35088462d1df88714ce2b52f4` | `main` | 2026-08-24T11:43:50+0600 | clean |
| Grounded-Segment-Anything | `Grounded-Segment-Anything` | `126abe633ffe333e16e4a0a4e946bc1003caf757` | `main` | 2026-08-24T11:44:00+0600 | clean |

`Acquired at` is the creation time of the clone's `.git` directory on this host, in local time (+0600). It records when the pin was taken locally — not when the upstream commit was authored.

---

## Remotes

| Repository | `git remote -v` (fetch/push, both identical) |
|---|---|
| SAM 2 | `origin  https://github.com/facebookresearch/sam2.git` |
| PixelHacker | `origin  https://github.com/hustvl/PixelHacker.git` |
| Moebius | `origin  https://github.com/hustvl/Moebius.git` |
| BrushNet | `origin  https://github.com/TencentARC/BrushNet.git` |
| ControlNet | `origin  https://github.com/lllyasviel/ControlNet.git` |
| InstructPix2Pix | `origin  https://github.com/timothybrooks/instruct-pix2pix.git` |
| Grounded-Segment-Anything | `origin  https://github.com/IDEA-Research/Grounded-Segment-Anything.git` |

Each clone has exactly one remote, `origin`, with identical fetch and push URLs. No additional remotes are configured.

---

## Upstream commit metadata

The date and subject of each pinned commit, as recorded in the upstream history. Useful for judging how current a pin is without re-cloning.

| Repository | Commit date (upstream) | Subject |
|---|---|---|
| SAM 2 | 2024-12-15T16:47:17-08:00 | remove `.pin_memory()` in `obj_pos` of `SAM2Base` to resolve and error in MPS (#495) |
| PixelHacker | 2026-06-20T16:29:37+08:00 | Update README.md |
| Moebius | 2026-08-12T11:16:32+08:00 | Add explicit license statement covering code and pretrained weights |
| BrushNet | 2024-12-17T21:49:54+08:00 | Merge pull request #77 from liyaowei-stu/main |
| ControlNet | 2023-09-09T12:09:12-07:00 | Update README.md |
| InstructPix2Pix | 2023-01-31T11:42:28-08:00 | Merge pull request #40 from 0xflotus/patch-1 |
| Grounded-Segment-Anything | 2024-09-05T14:07:32+08:00 | fix build model error (#526) |

---

## Submodules

Six of seven repositories declare no submodules (no `.gitmodules` present).

**Grounded-Segment-Anything** declares two, and both are **UNINITIALIZED** — `git submodule status` reports a leading `-` on each, and no submodule content exists on disk. Initialization is deliberately deferred.

| Submodule path | URL | Pinned gitlink SHA | State |
|---|---|---|---|
| `grounded-sam-osx` | https://github.com/linjing7/grounded-sam-osx.git | `6688b036c7856a302f9315bb16864d66fb2cdade` | NOT INITIALIZED |
| `VISAM` | https://github.com/BingfengYan/VISAM | `d7c38233882ff9d34d5cbecb8495e175e4dffc8c` | NOT INITIALIZED |

The gitlink SHAs above are read from the parent repository's tree. Because the submodules were never checked out, their contents were **not audited** and their licenses were **not inspected**.

---

## Weight / checkpoint state

**No model weights exist in any clone.** A recursive scan of all seven repositories for `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, `*.onnx` returned **0 matches**.

No download script was executed. Scripts present but deliberately not run:

- `sam2/checkpoints/download_ckpts.sh`
- `instruct-pix2pix/scripts/download_checkpoints.sh`
- `instruct-pix2pix/scripts/download_pretrained_sd.sh`
- `instruct-pix2pix/scripts/download_data.sh`

Documented checkpoint sources per repository are recorded in [`REPOSITORIES.md`](REPOSITORIES.md).

**A Hugging Face weights-repo revision is not a source commit.** Weight revisions, when acquired in later phases, are recorded in a separate field from these source SHAs and must never be substituted for them.

---

## Disk usage

| Repository | Total | of which `.git` |
|---|---|---|
| Grounded-Segment-Anything | 278 MB | 160 MB |
| ControlNet | 229 MB | 128 MB |
| PixelHacker | 223 MB | 192 MB |
| SAM 2 | 208 MB | 145 MB |
| Moebius | 165 MB | 113 MB |
| BrushNet | 78 MB | 39 MB |
| InstructPix2Pix | 36 MB | 17 MB |
| **Total** | **1.2 GB** | — |

Entirely source history — no weights, no environments, no build artifacts.

---

## Integrity verification

Re-derive every pin and compare against the table above:

```
cd /Users/atik/Projects/PixelForge/research/upstream
for r in sam2 PixelHacker Moebius BrushNet ControlNet instruct-pix2pix Grounded-Segment-Anything; do
  printf "%-28s %s  %-6s %s changed\n" \
    "$r" \
    "$(git -C "$r" rev-parse HEAD)" \
    "$(git -C "$r" rev-parse --abbrev-ref HEAD)" \
    "$(git -C "$r" status --porcelain | wc -l | tr -d ' ')"
done
```

Interpretation:

- **SHA differs** → the clone was updated, reset, or re-cloned at another commit. Every result derived from it is invalid until re-validated and the pin re-recorded.
- **Non-zero changed count** → the working tree was modified, violating the READ-ONLY rule. Inspect with `git -C <repo> status` and restore before proceeding.
- **Branch differs** → informational only. The SHA remains the authority.

Restoring a drifted clone to its pin (destructive to local edits, which should not exist):

```
git -C /Users/atik/Projects/PixelForge/research/upstream/<repo> checkout <SHA-from-this-file>
```

## Rules governing this file

1. This file is **derived**, never hand-edited. Regenerate it from the clones.
2. A SHA is written only after `git rev-parse HEAD` returned it from a real local clone, twice.
3. Tags and branch names are never authoritative.
4. Any change to a pin is a **provenance event**: update this file and [`REPOSITORIES.md`](REPOSITORIES.md) in the same commit, and re-run validation for the affected model.
5. Upstream clones themselves are never committed to this repository — only these registry files are.

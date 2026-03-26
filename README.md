# Conservative Test-Time Adaptation under Open-World Contamination via Binary Correctness Feedback

This repository is a clean public-release snapshot for the PPSN 2026 submission on conservative test-time adaptation under contamination. The central claim is narrow and deliberate: binary correctness feedback can act as an external veto that enlarges the empirically safer operating region under contamination, but this snapshot does not claim universal superiority of binary feedback over strong 0-bit test-time adaptation on stationary corruption benchmarks.

## Paper

- Paper title: `Conservative Test-Time Adaptation under Open-World Contamination via Binary Correctness Feedback`
- Snapshot version: `v1.0-ppsn2026`
- Final paper PDF in this release: [`paper/ppsn_submission_v3.pdf`](paper/ppsn_submission_v3.pdf)
- Online supplement PDF in this release: [`paper/ppsn_online_supplement_v2.pdf`](paper/ppsn_online_supplement_v2.pdf)

## Scope

This public snapshot is intended to:

- expose the implementation of the shared experimental harness and the compared methods
- preserve the processed paper-facing outputs used in the PPSN submission snapshot
- document how to inspect or regenerate lightweight summary artifacts from the included processed results

This public snapshot is not intended to:

- redistribute datasets
- redistribute large raw result dumps or benchmark archives
- bundle raw checkpoints or local machine caches
- claim that binary feedback universally outperforms strong 0-bit methods on stationary benchmarks

## What Is Included

- `src/`: implementation of the streaming adaptation methods and evaluation utilities
- `scripts/`: reproduction, analysis, and plotting scripts relevant to the snapshot
- `configs/`: YAML configs for training, frontier, contamination, robustness, and smoke workflows
- `tests/`: lightweight unit tests for fragile utilities and method behavior
- `results/processed/`: selected processed summaries and final CSV tables used in the paper
- `figures/final/`: final paper figures in PDF form
- `paper/`: the PPSN submission snapshot PDF, the online supplement PDF, and release-facing paper notes
- release-facing documentation such as `REPRODUCE_QUICKSTART.md`, `DATA_ACCESS.md`, and `PUBLIC_RELEASE_MANIFEST.md`

## What Is Not Included

- datasets and benchmark archives
- raw result dumps under `results/raw/`
- source checkpoints and caches
- external baseline repositories copied into the internal workspace
- local prompt logs, internal planning notes, and machine-specific audit clutter

## Setup

Python 3.9 is the intended baseline for this snapshot.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need GPU acceleration, install a PyTorch build compatible with your CUDA stack before or while installing the requirements.

## Quick Reproduction

The shortest realistic commands for the public snapshot are in [`REPRODUCE_QUICKSTART.md`](REPRODUCE_QUICKSTART.md).

At a high level:

```bash
python -m unittest discover -s tests -v
python scripts/build_tables.py --processed-root results/processed
python scripts/plot_figures.py --processed-root results/processed --figures-root figures/rebuilt
```

These commands validate the public code snapshot and regenerate lightweight tables or figures from the included processed outputs. They do not rerun the full benchmark matrix.

## Repository Structure

```text
github_release/
  configs/
  docs/
  figures/
    final/
  paper/
  results/
    processed/
  scripts/
  src/
  tests/
  README.md
  DATA_ACCESS.md
  REPRODUCE_QUICKSTART.md
  RELEASE_NOTES_v1.0-ppsn2026.md
  PUBLIC_RELEASE_MANIFEST.md
  PAPER_LINK_INSERTION.md
```

## Datasets

Datasets are not redistributed in this repository. See [`DATA_ACCESS.md`](DATA_ACCESS.md) for the required datasets, expected directory names, and clean acquisition guidance.

## Citation

If you use this snapshot, please cite the paper and the repository release metadata:

- repository metadata: [`CITATION.cff`](CITATION.cff)
- paper PDF: [`paper/ppsn_submission_v3.pdf`](paper/ppsn_submission_v3.pdf)

## Code / Release Status

- Status: PPSN 2026 submission snapshot
- Intended public tag: `v1.0-ppsn2026`
- Public repository name: `fbtta-ppsn2026`
- License: `MIT`
- Public repository URL: `https://github.com/dqswordman/fbtta-ppsn2026`
- Release URL: `https://github.com/dqswordman/fbtta-ppsn2026/releases/tag/v1.0-ppsn2026`

## Supplement DOI Placeholder

Once the supplement is archived, insert the DOI or persistent URL here and into the paper:

- Supplement DOI / URL: `<SUPPLEMENT_DOI_OR_URL>`

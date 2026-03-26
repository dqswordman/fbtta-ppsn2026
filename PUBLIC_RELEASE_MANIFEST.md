# Public Release Manifest

## Included

- `src/`
  - reason: core implementation of the methods, data loaders, metrics, models, and utilities
- `scripts/`
  - reason: public reproduction and analysis entry points
- `configs/`
  - reason: public configuration files for the reported workflows
- `tests/`
  - reason: lightweight verification of key utilities and method behavior
- `results/processed/paper_summaries/`
  - reason: selected processed summary CSV files that support the paper's reported results without shipping raw run dumps
- `results/processed/final_tables/`
  - reason: paper-facing CSV tables used in the PPSN submission snapshot
- `figures/final/`
  - reason: lightweight final figures referenced by the paper
- `paper/ppsn_submission_v3.pdf`
  - reason: final PPSN submission snapshot PDF
- `paper/ppsn_online_supplement_v2.pdf`
  - reason: final online supplement PDF
- `paper/PORTING_DISCLOSURE_TABLE.md`
  - reason: transparent shared-protocol porting disclosure for non-core baselines
- `docs/baseline_porting_notes.md`
  - reason: supplemental baseline-porting context
- top-level public-release documentation
  - reason: setup, citation, data access, release notes, and paper-link insertion guidance

## Excluded

- `data/`
  - reason: dataset redistribution is intentionally excluded
- `results/raw/`
  - reason: raw result dumps are larger, less curated, and not required for the public paper snapshot
- `checkpoints/`
  - reason: checkpoints are not required to understand the release and may be large
- `artifacts/`
  - reason: local caches and intermediate files are not appropriate for the public release
- `external/`
  - reason: third-party baseline repositories should be obtained from their official upstream sources
- `audit/`
  - reason: internal audit notes and machine snapshots are not part of the public release package
- root planning and prompt files such as `*_PLAN*.md`, `*_PROMPT*.md`, `AGENTS.md`
  - reason: internal workflow material, not public-release content
- old draft PDFs, LaTeX build products, and duplicate paper variants
  - reason: keep the public package small and unambiguous
- secrets, local environment files, and machine-specific paths
  - reason: security and cleanliness

## Publication status

- Public repository created: `https://github.com/dqswordman/fbtta-ppsn2026`
- Release created: `https://github.com/dqswordman/fbtta-ppsn2026/releases/tag/v1.0-ppsn2026`

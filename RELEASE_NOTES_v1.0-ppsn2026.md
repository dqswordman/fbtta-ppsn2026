# Release Notes: v1.0-ppsn2026

## Title

PPSN 2026 submission snapshot

## Summary

This release is a clean public snapshot of the code, processed paper-facing outputs, and documentation associated with the PPSN 2026 submission:

`Conservative Test-Time Adaptation under Open-World Contamination with an External Binary Correctness Veto`

## Included

- method implementations under `src/`
- reproduction, plotting, and analysis scripts under `scripts/`
- configuration files under `configs/`
- lightweight tests under `tests/`
- selected processed summary CSV files used to support the paper
- final paper-facing CSV tables under `results/processed/final_tables/`
- final paper figures under `figures/final/`
- the PPSN paper PDF and online supplement PDF under `paper/`
- release-facing documentation for setup, data access, citation, and paper link insertion

## Excluded

- datasets and dataset archives
- raw result dumps
- checkpoints and caches
- external baseline repositories copied into the private workspace
- internal planning logs and prompt material

## Scientific scope

This release snapshot supports a narrow scientific claim:

- binary correctness feedback can act as an external veto that expands the empirically safer operating region under contamination

This release snapshot does **not** claim:

- universal superiority of binary feedback on stationary corruption benchmarks

## Notes

- Public repository name: `fbtta-ppsn2026`
- License: `MIT`
- Supplement PDF location in the repository: `paper/ppsn_online_supplement_v2.pdf`

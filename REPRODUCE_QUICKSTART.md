# Reproduce Quickstart

This quickstart is for the public PPSN 2026 submission snapshot. It focuses on validating the released code and regenerating lightweight paper-facing artifacts from the processed outputs included in this release. It does not rerun the full heavy experiment suite.

## 1. Create an environment

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

Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need GPU execution for your own later extensions, install a PyTorch build compatible with your hardware and CUDA stack.

## 2. Validate the code snapshot

```bash
python -m unittest discover -s tests -v
```

## 3. Regenerate lightweight tables from included processed outputs

```bash
python scripts/build_tables.py --processed-root results/processed
```

This writes lightweight CSV and TeX tables under `results/processed/tables/`. Because this public snapshot does not include `results/raw/`, the optional compute table is skipped automatically.

## 4. Regenerate lightweight figures from included processed outputs

```bash
python scripts/plot_figures.py --processed-root results/processed --figures-root figures/rebuilt
```

This recreates lightweight summary figures from the included processed summaries. It does not rerun training or adaptation.

## 5. Inspect the paper-facing artifacts

- main paper PDF: `paper/ppsn_submission_v3.pdf`
- online supplement PDF: `paper/ppsn_online_supplement_v2.pdf`
- final CSV tables used in the paper: `results/processed/final_tables/`

## What this snapshot does not reproduce by itself

- dataset downloads
- end-to-end source training
- full frontier, contamination, or robustness sweeps
- reconstruction of the omitted `results/raw/` directory
- checkpoint-dependent reruns

For dataset requirements and expected directory names, see `DATA_ACCESS.md`.

# Repository working rules

## Current source of truth
- Current manuscript: `paper/current/main.tex` and `main.pdf`; supplement: `paper/current/supplement.tex` and `supplement.pdf`.
- Reviewer-response experiments: `experiments/reviewer_response/`.
- Public Volve reproduction: `reproducibility/public_volve/`.
- Historical papers and competition narratives are archival, not current instructions.

## Data and privacy
- Never commit raw internal train/test curves, labels, model weights, `*.npz`, `*.npy`, credentials or server caches.
- Historical submission JSON/ZIP files contain model predictions, not hidden ground truth.
- Public Volve CSV files remain ignored and are obtained from the upstream PDDA source.
- Run `python tools/sanitize_before_public.py` before publishing new host-specific material.

## Reproducibility
- Build paper with `latexmk -pdf -interaction=nonstopmode -halt-on-error` from `paper/current/`.
- Keep manuscript numbers, aggregate JSON results, figures and hashes consistent.
- Distinguish public reproducibility, internal OOF and historical blind platform records.
- Do not claim independent seed-wise ablation variance from legacy seed-averaged OOF arrays.
- Do not describe offline results as prospective physical or field validation.

## Changes
- Preserve user-authored revisions; use targeted fixes rather than reverting whole files.
- Review `git diff` before commit and never force-push to `main`.
- Regenerate `paper/current/manifest.json` and `RELEASE_MANIFEST.json` after changes.

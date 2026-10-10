# Paper draft v2 for Geodata and AI (Original Research Paper)

The compiling source is the LaTeX tree (server `paper_latex/`, output `main_v2.pdf`, 29 pages,
compiled clean). This markdown mirrors `main_v2.tex` content-wise: title, abstract, and all new
sections below match the typeset version. Figures are built by `solution_v7/fig_rebuild_1_8.py` +
`fig_new_9_13.py` on the server and are archived in `docs/figures/` (PNG 300 dpi + PDF) and
referenced by `docs/paper/`.

## Title

**Where the log ends: identifiability limits, tolerance-aligned modelling, and drift augmentation for reservoir-property prediction from wireline logs**

Short title: Drift, identifiability, and reservoir prediction

## Abstract

Predicting porosity (POR), permeability (PERM) and water saturation (SW) at every depth sample from wireline logs is increasingly evaluated by tolerance-based accuracy scores rather than squared error. On a 90-well benchmark (80 training wells, 730,268 samples; 10 blind wells) scored under a relative-tolerance metric, only 36–42 % of the sample-to-sample label variation proves recoverable from the thirteen input curves alone—the remainder is zonation and core-calibration information. Within that ceiling, tolerance-aligned Bayes-action decoding of binned predictive distributions, a jointly modelled reservoir-state variable, and a whole-well dilated CNN beat tuning and lift well-grouped cross-validation from 80.88 to 81.63; that progression transfers cleanly to the five blind wells (82.27 → 83.26), and ablation submissions show the contributions to be exactly additive. Type-well selection and DTW alignment, which won the SPWLA PDDA 2021 contest, do not transfer to this benchmark, and a transfer experiment on the public Volve data identifies the number of wells and the metric as the conditions under which each technique pays. Repeating the analysis on a second model family—a calibrated dual-backbone U-Net chain—exposes the second lesson: every in-distribution OOF gain of its newer 8-stage successor (up to +0.283) vanishes at the blind wells, which reward the older 18-stage chain (83.9329 versus 83.6076). PERM-only control swaps bound the leaderboard noise at ±0.04–0.08, so the regression is genuine and lives in the SW head's drift-sensitive default decisions. Per-curve KS distances to the training pool (0.114–0.240) match natural inter-well variation (LOO 0.107–0.440): blind wells are ordinary out-of-fold wells, not outliers. Damage then isolates to the backbone's SW default/continuous gate (honest out-of-fold sensitivity −1.61) rather than its continuous values, and a label-preserving per-well curve perturbation calibrated to the measured drift collapses the gate's sensitivity by about 95 % (−0.69 → −0.03). A transplant keeping the backbone's continuous values and swapping only that gate gains +1.36 on SW under synthetic drift, yet does not pay: nine further controlled submissions—PERM-only head swaps, posterior fusions, majority votes, single-seed and ratio-tuned variants, the transplant—all fall 0.0–0.38 points below the unchanged baseline, showing that at this resolution the blind set rewards the baseline prediction's exact morphology. The lesson is joint: identifiability bounds what logs carry, tolerance-aligned decoding is required to read it, division of labour beats one-model thinking, and well-level drift is where distribution-aware care outperforms careful models. Code, machine-readable evidence sets, and all per-version prediction artifacts are released.

## Keywords

well logs; petrophysics; gradient boosting; sequence model; decision theory; domain shift; drift augmentation; benchmarking; reservoir characterisation

## Highlights (3–5 bullets, ≤ 85 characters each)

- Interpreted POR/PERM/SW labels contain information absent from the input logs.
- Bayes-action decoding of binned distributions matches tolerance-based metrics.
- Whole-well dilated CNN segments reservoir states better than pointwise trees.
- Blind wells absorb in-distribution OOF gains (+0.05–0.58) entirely, 9 times.
- Label-preserving curve perturbations collapse drift sensitivity by 95 %.

## 1. Introduction  *(unchanged from v1 draft)*

## 2. Data and evaluation

### 2.1 Well-log benchmark & 2.2 Official metric & 2.3 Validation protocol

*(unchanged from v1 draft)*

### 2.4 Distribution shift between training and blind wells

The 10 blind test wells and the 80 training wells share the same field setting but differ at the level of raw curves. For every blind well we measure, per curve, the Kolmogorov–Smirnov distance to the pooled training distribution and the median shift in pooled-IQR units (Fig. 1). Across the nine log curves, the mean per-curve KS distance ranges from 0.114 to 0.240; median shifts reach 0.55 IQR on CAL, 0.48 on RT and 0.28 on PE—the curves exhibiting the strongest borehole, resistivity and tool-baseline differences.

To test whether this profile is unusual, we repeat the same comparison against the training pool itself: every training well, contrasted to the pool of the other 79, exhibits a mean KS of 0.107–0.440 with median 0.181. The blind-well profile (0.114–0.240, median ≈ 0.169) therefore sits squarely inside the natural between-well variation of the field. The lesson is practical: no exotic distribution shift requires special treatment, and cross-validation cannot anticipate blind-set behaviour from within ordinary field diversity.

## 3. Methods

### 3.1–3.7  *(unchanged from v1 draft: baselines, lag/zone features, distribution models and Bayes decode, joint states, dilated CNN, nested fusion, well similarity and DTW alignment features)*

### 3.8 The calibrated dual-backbone deep family

The second model family studied in this work is a deep, multi-stage chain trained with per-point conditional affine calibration: a two-branch U-Net backbone (a 1,024-sample full-well branch and a 256-sample local-lag/within-well branch) whose outputs are decoded by per-posture affine scalers initialised at the identity. The training sequence updates a nested set of sub-networks over progressively enlarged objectives: posteriors, prior posteriors, behavioural mixture posterior, label-mixture posterior, and finally the utility-decoding mixture over a fixed 800-point action grid and default atom—decoded by choosing the Bayes action expected under the mixture posterior.

At inference the five fold-specific bundles vote with expected-utility fusion, the chain being re-applied at successive stages with varying action and calibration scopes. Within this family two complete chains were produced and submitted throughout the competition: v97 (18 stages) with OOF 81.891 and blind leaderboard 83.9329, and v370 (8 stages × 2 seeds) with OOF 81.990 / 82.174 / 82.195 (per-seed setup) and blind scores 83.5547 / 83.6076 / 83.6044.

## 4. Identifiability audit *(unchanged from v1 draft: ceiling 36–42 %)*

## 5. Results

### 5.1–5.6  *(unchanged from v1 draft)*

### 5.7 When OOF increases lose at the blind wells: the v370 paradox

Quantify the v370-vs-v97 paradox: the newer chain outscoring the baseline by +0.283 OOF but losing by −0.325 at the blind wells (83.6076 versus 83.9329). First bound the leaderboard noise floor via three PERM-only head swaps (OOF +0.043 to +0.050; blind deltas −0.078, −0.004, −0.036), showing a measurement noise floor of about ±0.04–0.08; the full-family degradation is 4–8× that floor, hence genuine. Row-level decomposition shows the damage to be carried by the SW head (2,171 default↔continuous flips + 2,021 rows beyond ±5 % tolerance) over POR (907 + 712) and PERM noise. Distribution over the ten blind wells (Fig. 9) localises the worst SW damage at the highest-drift wells, d328607c accumulating 422 SW band crossings under a mean KS of 0.240.

## 5.8 Drift augmentation: repairing the SW head with label-preserving curve perturbations

Each training well is augmented with K = 2 label-preserving copies perturbed by per-curve shifts drawn in pooled-IQR units (σ per well from U(0.03, 0.20), curve-weighted ×1.3–1.6 for CAL/RT/PE, plus 4 % scale jitter and a small log-domain jitter on RT/RXO), keeping POR/PERM/SW labels unchanged. Perturbation magnitudes are calibrated so that synthetic wells reproduce the measured blind-well drift (mean KS 0.114–0.240), which also matches the natural LOO inter-well variation of the training pool itself. Two-arm ablations with the same head recipe (CatBoost SW gate + regressor, and a 60-bin distributional variant) show clean OOF almost unchanged (74.84 vs 74.74) but drift sensitivity collapsing from −0.69 to −0.03 (−96 %) and −0.57 to −0.03 (Fig. 11). Under the same perturbations the augmented gate reaches 74.71 vs 74.15 (unaugmented twin) and 73.93 (the backbone's own SW head under per-fold-restricted honest predictions).

Decomposing further, keeping the backbone's SW continuous values and swapping only its default/continuous decisions ("B′") yields clean 75.31 / drift 75.29: +1.36 over the baseline head under drift. The assembled B′ transplant submission (5,880 of 95,948 rows changed) did not pay on the blind leaderboard, feeding §5.9.

## 5.9 What the blind wells reward: nine controlled submissions

Pooling the PERM-only probes, fusions (v4, v5), the 2-veto vote (v6), the standalone DL variants (v6b, v6c, v370-equal) and the B′ transplant yields nine submissions each with a positive in-distribution OOF gain over the baseline (range +0.043 to +0.58) and a blind deficit of 0.0–0.38 (Fig. 12). Two robust patterns: (i) only the minimal-morphology variant (2.2 % of rows, 2,144 rows) reaches the noise band of a tie, all other changes pay roughly monotone in the number of changed rows; (ii) OOF improvement margins in this regime have zero blind-set value.

## 6. Type-well selection and DTW alignment on the 80-well benchmark *(unchanged from v1: does not help)*

## 7. Transfer to the SPWLA PDDA 2021 Volve benchmark *(unchanged from v1: 9 wells & Brier metric)*

## 8. Discussion

Within-family, needle-in-haystack gains of structure are real and track to the blind wells: identifiability ceiling 36–42 %, and the three structural layers (Bayes decode, joint states, whole-well CNN) lift cross-validation 80.88 → 81.63 (additive ablations +0.720, +0.266 exactly additive); type-well selection and DTW alignment do not help here but have their own transfer conditions quantified on Volve. Across-family, four times the measured blind-set noise floor separate the older and newer DL chains (−0.325 with ±0.04–0.08 floor): every OOF-positive repair we submit fails at the blind wells, even though the wells are ordinary out-of-fold wells in input space; blind-set credit accrues to the older chain's prediction morphology, and only the minimal-change variant reaches a tie within the noise band. The drift-augmentation method motivated by the diagnosis does work: label-preserving per-well perturbations collapse the SW gate's drift sensitivity by about 95 % (−0.69 → −0.03) at a cost of only 0.1 clean-OOF points, and the B′ gate transplant measures +1.36 on SW under the same perturbations—the largest quantified in-class SW gain of the project, simply not what a five-well leaderboard of well-level similarity prices.

## 9. Conclusions

On a 90-well wireline-log benchmark scored by a relative-tolerance metric, only 36–42 % of the interpreted-label variation is recoverable from the input curves alone (identifiability ceiling), and within it structural decisions beat tuning: Bayes-action decoding, joint states and whole-well CNN lift cross-validation from 80.88 to 81.63 and blind wells from 82.27 to 83.26, with additive ablation deltas (+0.720 gates, +0.266 values). Type-well selection, DTW alignment and Volve-transfer experiments complete the first model family's picture: these techniques have their own conditions, and this benchmark is not among them.

Across-family the second model family shows the same programme failing in a second way: once blind-set resolution matters, the leaderboard defends morphology rather than ranking in-distribution improvements. Every measured OOF gain of the newer chain—architecture, seeds, weights, votes, fusions and finally a drift-repair of the exact component diagnosed as the failure carrier—does not pay at the five blind wells, against a measured blind noise floor of ±0.04–0.08 and in-distribution gains of up to +0.58. Since the blind wells are ordinary out-of-fold wells in input space (§2.4), the bottleneck is not domain outliers: blind-set credit scores the rows whose prediction morphology matches the current leader.

The project also contributes a directly usable method: label-preserving per-well curve perturbations calibrated to measured well-level drift, which collapse the drift sensitivity of the vulnerable model head by about 95 % at a cost of roughly 0.1 official clean-OOF points—the first step we take whenever blind-set drift is measured, before deciding to train, transplant, or simply preserve. Code, machine-readable evidence sets and all per-version prediction artifacts are released.

## Figure and table index

- **Fig. 1** — Blind-well drift against the training pool (`fig9_drift_profile`, style V2)
- **Fig. 2** — Where can the labels be learned from? (`fig1_identifiability`, style V2)
- **Fig. 3** — Baseline gate errors concentrate at boundaries; thin beds are missed (`fig2_gate_anatomy`)
- **Fig. 4** — Regression vs segmentation by model family (`fig3_division_of_labour`)
- **Fig. 5** — Two generations and two lessons: CV keeps rising, blind wells defend the older chain (`fig4_progression`, extended with the DL generations)
- **Fig. 6** — Per-property cross-validation accuracy by generation (`fig7_generations_per_target`)
- **Fig. 7** — Held-out well tracks: v1/v2/v3 (`fig6_example_well`)
- **Fig. 8** — Blind leaderboard: three generations, the v3 ablation and the second model family (`fig5_leaderboard_ablation`, extended with DL v97 and v370-two-seed)
- **Fig. 9** — The newer chain's SW changes target the drift profile (`fig10_perwell_damage`)
- **Fig. 10** — PERM-only head swaps bound the blind noise floor at ±0.078 (`fig11_perm_control`)
- **Fig. 11** — Label-preserving curve perturbations collapse drift sensitivity by ~95 % (`fig12_drift_augmentation`)
- **Fig. 12** — Nine controlled submissions: every OOF gain vanishes inside or below the noise band (`fig13_submissions_chain`)
- **Fig. 13** — Type-well selection (leave-well-out) (`fig8_typewell`)
- Table 1 — Data table (unchanged)
- Table 2 — Audit table (unchanged)
- Tables 3–5 — Nested fusion / nets / results (unchanged)
- Table 6 — Blind leaderboard: three generations and four ablations (unchanged)

## Data availability

Repository mirrors all results cited in this paper. Well-grouped out-of-fold predictions, per-well predictions, nested fusion outputs, drift characterisation tables and the drift-augmented ablation runs are under `results/` (tree–sequence family), `results/dl_fusion/` (DL family: E1–E3 weight grid, nested CV, per-row dump), `results/drift_sw/` (drift augmentation: two-arm ablations, calibration mappings, sensitivity and transplant details) and `results/dl_fusion/drift/` (test-versus-train and inter-well LAO drift descriptors). Blind leaderboard tags are per-submission in each `submissions/v*/manifest.json`. This paper's figures are rebuilt with `solution_v7/fig_rebuild_1_8.py` and `solution_v7/fig_new_9_13.py` against those evidence sets; CSV snapshots of each figure are kept in `docs/figures/`. All inference builds are deterministic on fixed hardware and sealed by SHA256 manifests (RUNTIME_PINNING.json). The test wells of the leaderboard evaluation were reported by the users after submission and are not independently verifiable by the authors on the platform at this time.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

*(unchanged from v1)*

## References

*(unchanged from v1; verify every DOI before submission.)*

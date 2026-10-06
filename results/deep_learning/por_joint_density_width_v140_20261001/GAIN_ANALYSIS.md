# POR head gain analysis

Original fixed fifth only; frozen predictions, no fitting or new selection. Single fold/seed and unequal active capacity limit causal attribution.

| Comparison | POR delta | Positive wells /16 | Nondefault delta | Default delta |
|---|---:|---:|---:|---:|
| local_vs_point | -0.000872 | 7 | -0.024372 | +0.011658 |
| weight_added | +0.167638 | 10 | +0.696735 | -0.114459 |
| width128_vs64 | +0.140076 | 10 | +0.603565 | -0.107040 |
| gate_added | -0.188560 | 5 | -0.790689 | +0.132476 |
| overall_vs_parent | +0.328636 | 10 | +1.394254 | -0.239516 |
| width256_vs128 | +0.043013 | 10 | +0.010384 | +0.060409 |
| overall256_vs_parent | +0.371649 | 11 | +1.404638 | -0.179107 |

Interpretation: point-local comparison provides no local-depth benefit. Adding conditional logweights alongside locations and then widening the nonlinear head each helped this exact POR recipe. Frozen original predictor/default gate/scales constrain the change to conditional mixture locations and masses. Extra output/capacity and different training trajectories prevent claiming isolated weight-only or width-only causal effects beyond this bounded comparison.

Next: width256 from same frozen v113 initial function, replace width128 head phase; do not warm-start it and add a sixth phase. Preserve original proper mixed-family NLL/CDF and scorer. Only after useful POR result, test the same readout mechanism on PERM and SW with matched controls. PERM width128 previously +0.009542 but width256 negative, so transfer is not automatic; SW Student3 is a separate family experiment and does not test this head mechanism.

Per-well/range details are in gain_analysis.json, for diagnosis only; no per-well/row model selection or label-derived input features.

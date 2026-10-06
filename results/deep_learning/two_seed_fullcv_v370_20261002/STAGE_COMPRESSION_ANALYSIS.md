# Stage compression evidence

Actual adjacent-stage changes on completed single-fold models; folds[1, 2, 3, 4] and paired seeds are not independent replicates. Historical fifth has differing budgets and is not pooled here.

Near-zero threshold0.0001 is a reporting convention, not a significance test. Actual raw values remain in STAGE_GAINS.csv.
The current fixed8-stage recipe is unchanged. The table does not estimate full OOF or causal stage-removal effects.

| Stage | Positive / near-zero / negative | Target gain min / mean / max | Mean TOTAL increment | CONT positive / negative |
|---|---|---|---:|---|
| POR113 | 7 / 0 / 1 | -0.011892 / +0.115707 / +0.277317 | +0.034712 | 6 / 2 |
| POR151 | 3 / 5 / 0 | +0.000000 / +0.042840 / +0.172053 | +0.012852 | 3 / 0 |
| PERM154 | 3 / 5 / 0 | +0.000000 / +0.018518 / +0.073516 | +0.006481 | 3 / 0 |
| SW104 | 6 / 0 / 2 | -0.187626 / +0.382934 / +0.715437 | +0.134027 | 7 / 1 |
| POR341 | 7 / 1 / 0 | +0.000000 / +0.030998 / +0.086374 | +0.009299 | 0 / 7 |
| PERM336 | 5 / 3 / 0 | -0.000008 / +0.051113 / +0.128419 | +0.017890 | 1 / 4 |
| SW351 | 6 / 2 / 0 | -0.000000 / +0.023437 / +0.092658 | +0.008203 | 5 / 1 |

No verified lossless8-to5-stage reduction follows from these changes. Subsequent stages depend on earlier trained states, and probability-mixture decoding is nonlinear.

Terminal stages in still-incomplete models (not pooled above):


Next isolated fixed-fifth comparisons, after fullCV:

- POR151 controlled omission with downstream POR341 retrained from POR113, same source/budget/seed; first7-stage candidate only. Positive completed or pending-stage gains forbid assuming universal redundancy.
- POR341 modification or omission: its observed benefit shifts DEFAULT/CONT; measure both and final native mixture, not solo score only.
- PERM154 omission needs neutral adapter and PERM336 retraining; PERM336 has meaningful fold2 gains, not universally redundant.
- SW104 source-preserving additive alternative within the existing stage slot; observed SOURCE regressions 2/8, positive cases 6; no method-wide failure claim.
- SW351 cannot be assumed redundant: 6 of 8 current observations positive. Any5-stage claim needs an actual fixed-fifth controlled recipe.

Complete already-authorized original+seed13 fullCV serially first. No new pruning/repair experiment launched.

Evidence: completed *_PREFIX_SNAPSHOT.json / *_TERMINAL.json and STAGE_GAINS.csv.

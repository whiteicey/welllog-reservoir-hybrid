# Continuous-row action diagnosis

Saved predictions only; no training, weight change or rowwise choice. Original continuous labels define analysis rows.
Predicted DEFAULT/CONT uses the original action threshold, not an inspected internal gate.
Group contributions sum exactly to each recorded CONT delta. Internal gate causality remains unproven.

| Fold | Model | Stage | CONT delta | Predicted DEFAULT before/after | CONT-to-CONT contribution | CONT-to-DEFAULT contribution |
|---:|---|---|---:|---:|---:|---:|
| 1 | original | POR341 | +0.000000000 | 7856/7856 | +0.000000000 | +0.000000000 |
| 1 | original | PERM336 | -0.000037185 | 5379/5379 | -0.000037185 | +0.000000000 |
| 1 | seed_20261013 | POR341 | -0.365965588 | 7981/8415 | -0.034841971 | -0.331123617 |
| 1 | seed_20261013 | PERM336 | -0.154116381 | 5229/5361 | -0.048338770 | -0.105777611 |
| 2 | original | POR341 | -0.411758405 | 8404/8994 | +0.008091064 | -0.420123746 |
| 2 | original | PERM336 | -0.065117387 | 4655/4827 | +0.165303504 | -0.245346621 |
| 2 | seed_20261013 | POR341 | -0.420131346 | 7993/8708 | +0.056786493 | -0.476917839 |
| 2 | seed_20261013 | PERM336 | -0.120109005 | 4649/4778 | +0.055122917 | -0.186157738 |
| 3 | original | POR341 | -0.236670540 | 8984/9379 | +0.138293931 | -0.379706207 |
| 3 | original | PERM336 | -0.258582490 | 6328/6507 | +0.000206281 | -0.265274233 |
| 3 | seed_20261013 | POR341 | -0.405248785 | 9036/9531 | +0.076057466 | -0.483231697 |
| 3 | seed_20261013 | PERM336 | +0.229528136 | 6902/7025 | +0.376163287 | -0.158416760 |
| 4 | original | POR341 | -0.347109579 | 7327/7835 | +0.065259092 | -0.412368671 |
| 4 | original | PERM336 | +0.000060359 | 4639/4639 | +0.000060359 | +0.000000000 |
| 4 | seed_20261013 | POR341 | -0.317041224 | 7654/8135 | +0.065065687 | -0.382106911 |
| 4 | seed_20261013 | PERM336 | +0.000010924 | 4911/4911 | +0.000010924 | +0.000000000 |

Full four-group breakdown is in CONTINUOUS_ACTION_DIAGNOSIS.json.
Cases may share folds/training wells; this is descriptive evidence, not independent replication or a causal ablation.

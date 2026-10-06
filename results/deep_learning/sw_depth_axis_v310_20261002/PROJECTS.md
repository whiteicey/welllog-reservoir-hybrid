Goal: improve continuous SW representation without losing retained whole score. Completed: research, matched residual implementation, source/phase plan. Next: explicitly launch depth/channel pair; record whole/DEFAULT/CONT, real timing and verdict. Not done: training, independent disk reload, remaining folds, fullOOF>82. Optional after verified gain: actual eight-stage composition; no automatic promotion.

Explicit pair launched: depth PID85880, channel PID76952; max2 normal GPU; gains pending real results.

Running snapshot (not final gain): {"depth": {"status": "running", "pid": 85880, "stage": "depth", "epoch": 8, "best": 74.1349342495239, "best_epoch": 5, "bad": 3, "peak_MiB": 1390.0986328125}, "channel": {"status": "running", "pid": 76952, "stage": "channel", "epoch": 7, "best": 74.14086578545242, "best_epoch": 4, "bad": 3, "peak_MiB": 1714.6171875}}; both below retainedSW74.16003022301975; current optional8TOTAL80.69562059394032 unchanged. Training continues under own whole-SW best and patience14.

V310 depth: SW74.160199923047, actual TOTAL80.625931901205; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

V310 channel: SW74.155625583198, actual TOTAL80.624330882258; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

Final310 whole scores {'depth': 74.16019992304726, 'channel': 74.15562558319847}; depth-channel +0.004574339849; CONT paired -0.013838135467; strata {'depth': {'default': 98.84885176097337, 'continuous': 17.452652710549867}, 'channel': {'default': 98.83626121843403, 'continuous': 17.466490846016587}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 649.342s; no fullOOF or remaining folds.

310 closed: depth-channel whole+0.004574339849/CONT-0.013838135467; depth versus297 whole+.000169700028/CONT-.050100056797. Actual8 diagnosticTOTAL80.695679988950, delta+0.000059395010, strict808false, not adopted for continuous focus. Retained8TOTAL80.69562059394032 unchanged; evidence/checkpoints positive/control preserved, no generic kernel sweep. Pairwall649.342s, depth632.700s/channel644.247s; two-process timing not serial ranking.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

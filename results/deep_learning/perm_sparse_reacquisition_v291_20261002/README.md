# PERM sparse curve reacquisition v291

## Focus
TabNet-inspired two sparsemax curve selectors and GLU representation updates,207 dense bypass and direct categorical readout. Joint new encoder/readout; original154 prior/gate frozen. Gamma1.3,48state, two blocks; selector output zero-initialized, paired initial uniform masks. Curve positions from source selected_feature_names, not guessed. Own PERM labels only.

## Explored and verification
Prepared, untrained. Point versus valid-window mean selector logits is the sole paired change. Same parameters, seed, initialization, loss/direct component CE1/fixedalpha.3 ramp5/Q/scoring/max80/pat14. Each explicit launch uses one added PERM phase; final method max8, no previous repair weights. Historical283 has different capacity and auxiliary readout count, not a strict mechanism ablation. Source paper/code and limitations in ../cross_domain_research_20261001/SPARSE_FEATURE_REACQUISITION_RESEARCH.md.

## Gain
Unproved. Absolute reference PERM28189.498908539006, prior15489.405564856919. Final primary delta point minus window; validate best-restored predictions and actual three-target composition. Live scores are snapshots, not completed gain.

## Not worth continuing
No evidence yet. If point fails paired window control, close this point-selection configuration without gamma/blocks grid. Failure does not invalidate all TabNet mechanisms. Track training active support/mask variation to distinguish inactive selector; these are not causal importance.

## Timing
Record individual training and batch wall time, concurrency and peak memory. Trainer time includes data preparation, initial evaluation and final restoration, code preparation and later CPU assembly excluded. No GPU numerical preflight.

V291 final PERM89.515554793257, gain+0.109989936338, actualTOTAL80.650971895712; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds231.108907/peakMiB788.697754. No otherfolds.

V291point finalPERM89.515554793257/best5/stop19, vs281+.016646254251. Actualoptional8TOTAL80.691306394366, all144665/frozen/RAMbestreplay verified, notdiskreload. Windowpairedfinalpending, no causalpairgain claim yet. New36466encoder/readout joint,8phases/eachhead+1, strict808false/nootherfolds/fullOOF82unmet. POR292 explicit launched original151/share50, no290weights.

V291 final PERM89.510323066916, gain+0.104758209997, actualTOTAL80.649140791493; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds216.946259/peakMiB788.697754. No otherfolds.

V291paired final point89.515554793257 vswindow89.510323066916, fairdelta+0.005231726341; bothbest5/stop19/same36466params/initial89.405566626894, jointbody/readout grads and training masks changed. Adoptpoint, +.016646254251 vs281. Actual8TOTAL80.691306394366, strict808false/nootherfolds. Pair isolates selection granularity only, not individual sparsemax effects. Bothpositive/control weights kept. Timing{"first_process_started_utc": "2026-10-01T16:42:03.718643+00:00", "point_started_epoch": 1790872923.7186427, "window_started_epoch": 1790872989.5238235, "concurrency": 2, "maximum_GPU_jobs": 2, "terminal": true, "batch_wall_start": "first OS process creation; includes data prep and warmup", "code_preparation_excluded": true, "batch_wall_seconds": 290.45420360565186, "individual_seconds": {"point": 231.1089074999909, "window": 216.94625869998708}, "peak_MiB": {"point": 788.69775390625, "window": 788.69775390625}, "completed_utc": "2026-10-01T16:46:54.172846+00:00", "wall_end": "last training report write mtime; CPU composition excluded", "startup_overlap_partial": true}. Atlasbatch updated.

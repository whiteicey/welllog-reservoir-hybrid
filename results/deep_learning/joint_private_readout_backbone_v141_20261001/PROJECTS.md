# v141 new joint backbone with private nonlinear target readouts

Focus: test whether distinct target-specific nonlinear representations improve the positive aligned random student. Return to v132r1 dense207/local/global-attention backbone architecture, replace its single linear75 output with three private96->192GLU96->LayerNorm/GELU->25 branches. Separate prediction mappings for POR/PERM/SW, shared encoder and own three-target losses. This is an architecture/capacity comparison, not proof that gradient interference is the cause.

Explored/verified: source/trainer/record helper prepared, not launched. Zero branch output weights and shared75 prior bias retain identical initial distribution priors to control. No trained v132r1 or v137 student weights loaded; all student weights random. Frozen original own64-fold teacher hidden alignment0.1 only in training/common128 core, teacher native256/student512. Teacher predictions never enter loss or inference; no pseudo-labels. Same original9curves/206/relative/train-only stats/Q50/splits/seed/action grids/CDFs/proper3-label likelihoods.

Gain: untrained/no score. Control actual fifthTOTAL79.240236161948, perchannel74.699142364354/89.114178052716/73.258660383403. New private readouts add parameters; no equal-capacity claim. Actual three-output TOTAL selects checkpoint. Need final all144665x3 best replay and frozen teacher validation, not partial initial-learning gain.

Not worth continuing: failed private ConvGRU/plainlabel/curve-token joint configurations closed with scoped evidence. Current pyramid v137 still active; do not assume its final result. No readout-width/depth/LR grid this batch absent useful gain.

Pending: launch explicitly only after current backbone v137 terminal and process handle absent. Max2GPU alongside channel route, normal/80-18/min10/b64/64steps/6e-4/cos80*64; atomic complete last and explicit resume, temporary process-only keep-awake, record real wall/overlap/peak. Standalone2dependency phases including original source teacher/Q, potential private-channel recipe5. No remaining folds before strictfifthTOTAL>80.8+gain, fullOOF>82 unmet; historical delivery unchanged.

Optional: retain useful private prediction architecture or return to new temporal communication if final comparison fails.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

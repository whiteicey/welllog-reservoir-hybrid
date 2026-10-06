# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW sparse curve reacquisition v294

## Focus
Transfer PERM291's positive point-vs-window selection mechanism (fair+.005231726341) to SW encoder, not its trained weights. New207dense bypass/two9curve sparsemax selectors/GLU48 updates, gate and categoricaldensity readouts jointly trained. Original104 prior frozen, calibratedtrainQ BCE/statebound4/share.25ramp5/conditionalRPS1/directcontinuoushardsoftCE1 retained. One added SW phase, no289/291repairweights.

## Explored and verification
Prepared/untrained. Same36515activeparams and seed20261294 for point/window; zero selector final layer/uniform masks, gatezero and densitynonzero init. Only selectorlogits point vs validwindowmean. Padding excluded and no crosswell. Metadata source exact9curveindices, no guesses. Balanced criterion is explicitly enforced for BOTH named selection arms; stage name cannot switch to ordinaryBCE. DirectauxCE uses final representation only, so historical289 is absolutecontrol, not strict capacity/auxcount ablation. Own SW labels only, no releasedweights/teachers/testlabels.

## Gain
Unproved. Predefined paired delta point minus window; absolute compare retained28974.1537264289993 and continuousstronger25274.14922820781207. Finalrestore/actual144665 assembly and default/continuous strata needed; overalldefault gain alone is not continuous improvement. Max80/pat14/min10/b64/64steps/cos80 unchanged/max2GP

22997928740; learned gain+0.030064454806, vs211+0.004194521948. Six optionalphases, default5; new36515sparse_rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V294point finalSW74.151817144575/best6/stop20, vs289-.001909284424; strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW294point": {"default": 98.86811932865155, "continuous": 17.380759647269663, "overall": 74.15181714457486}}. Actual144665/RAMbest/frozen verification, no diskreload. Windowongoing, pairedfinalpending; noadoption/cleanup beforecomparison. POR295point explicitlylaunchedPID78300 withwindowheld, companionSW294windowPID21088/max2GPU.

V294 window final SW74.145463813500, actualTOTAL80.620774262864; learned gain+0.023711123731, vs211-0.002158809127. Six optionalphases, default5; new36515sparse_rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V294paired final point74.151817144575/best6/stop20 vswindow74.145463813500/best14/stop28, same36515params/initial74.121752689769/80-14 budget, delta+0.006353331075. Bothbelowretained289; strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW294point": {"default": 98.86811932865155, "continuous": 17.380759647269663, "overall": 74.15181714457486}, "SW294window": {"default": 98.77535448509161, "continuous": 17.572885559050093, "overall": 74.14546381349975}}. Pointcontinuous worsens, noadoption/noSWcontinuous breakthrough. Keepfairgranularitypositive/control checkpoints and predictions; close standalone sparseSW route. Actual144665assembly/RAMbest/frozen checks, notdiskreload/fullOOF. Batchtiming{"point_process_start_utc": "2026-10-01T16:51:45.825622+00:00", "point_start_epoch": 1790873505.825622, "maximum_concurrent_GPU": 2, "terminal": true, "includes_prep_initial_train_final": true, "excludes_codeprep_CPUcomposition": true, "window_start_epoch": 1790873577.4945474, "concurrency": 2, "not_isolated_speed_comparison": true, "batch_wall_seconds": 462.9522874355316, "individual_seconds": {"point": 256.6073596000206, "window": 383.6069831000059}, "peak_MiB": {"point": 1147.96630859375, "window": 1145.0009765625}, "partial_overlap": true, "wall_end": "last trainingreport write; no codeprep/CPUcomposition"}.



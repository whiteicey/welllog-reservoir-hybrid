# v138 private PERM density width256

Exploration focus: widen the nonlinear point-conditioned location/logweight readout from128 to256 on the same frozen original PERM source. Current v136 width128 shows preliminary89.387936, +0.009542 vs compact control, not yet final. Launch only after final evidence confirms positive control gain.

Explored/verified: prepared new source and trainer, no training yet. Original source predictor fully frozen, 319 inputs (two detached128 expert features +63 amplitude inputs), hidden256/GLU512/32zero outputs; stored33 point-summed depth coefficients. Same Gaussian NLL, means bounded by one fixed reference sigma and logweights +/-1. Same zero-output initial source function89.344760174753, no trained v136 readout loaded. Original9curves/206features/train-only stats/Q50/fixed fifthseed and decoder unchanged. This is a capacity comparison, not same active parameter count.

Gain: untrained/no score. Compare final PERM with width128 v136 and current v10089.378394211144, plus actual three-target TOTAL with fixed v134POR/v104SW. Initial learning is not mechanism gain; validate restored best144665 points/frozen parameters once final.

Not worth continuing: no gate or density-family changes and no LR grid this batch. Prior fixed heavy-tail and linear mean opening negative; conclusions specific to those designs. No repeated post-hoc decoder or threshold tuning.

Pending: explicitly launch only after PERM v136 completes and demonstrates gain, max2GPU with backbone route.80/18/min10/b64/64steps/3e-4/cos80*64; atomic complete last state and explicit resume; temporary keep-awake, measure launch wall/trainingseconds/overlap/peak. This replaces the PERM training phase: original source +POR113+POR134+PERM138+SW104 =5 total phases; not source+PERM136+PERM138. Strict fifthTOTAL>80.8+gain before other folds; original fullOOF>82 unmet, deliveries immutable.

Optional: retain width only if actual gain; do not continue width scaling after a negative result without new mechanism evidence.

V138 explicitly launched, actual PID62004 command verified; snapshot {"status": "running", "pid": 62004, "stage": "wide256", "epoch": 7, "best": 89.3447601747529, "best_epoch": 0, "bad": 7, "peak_MiB": 786.46533203125}. Training partial, no gain yet. This supersedes prepared-only text. Positive v136 control retained; no warm-start, final recipe still5 phases. Parallel with v135; max2GPU.

V138 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.574006662593, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=151.874, peakMiB=786.465; all144665replay/frozen exact.

Final v138 PERM89.344760174753 best0/stop18, versus width128 v136 -0.043175921031 and compactcontrol -0.033634036391. No gain; close width expansion, retain v136 positive model. Negative best/last removed with SHA manifest after actual assembly/replay.
Timing {"training_seconds": 151.87421049998375, "launch_to_report_wall_seconds": 157.391062, "peak_MiB": 786.46533203125, "overlap_other_route": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}; removedMiB=22.19795799255371. No remaining folds, originalOOF>82 not achieved.

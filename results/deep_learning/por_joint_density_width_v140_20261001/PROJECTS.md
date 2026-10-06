# v140 POR conditional density width256

Focus: extend the measured POR width64->128 benefit (+0.140076259170) with one bounded width256 comparison. Positive width128 POR77.869090371913; current actual recipeTOTAL80.589118234954. Same frozen v113 mixed Student3/Cauchy predictor, joint location/logweight update, 319 inputs and point-only effective kernel. New headwidth256/GLU512/32 outputs, zero initial output. Capacity change only, not same active parameter count.

Explored/verified: prepared source/trainer/record helper. No weights from trained v134 head loaded; source+v113 frozen parent defines same initial predictive function77.540454326739. Random new hidden parameters differ. Own POR proper NLL/BCE, referencewidth-bounded locations +/-1 fixedreference sigma, logweight +/-1; gate/scale/old predictors frozen. Original9curves/206+63amplitude/train-only stats/Q50/fixed fifth/seed/scoring/CDF/action grid unchanged. Cauchy mixture has no finite overall mean/variance; referencewidth is not whole-mixture SD.

Gain: untrained/no score. Compare final best POR with width12877.869090 and actual three-target assembly with fixed positivePERMv136/SWv104. Do not claim initial-function or partial improvement as final capacity gain. Original strictTOTAL>80.8+gain and completeOOF>82 are not achieved.

Not worth continuing: PERM width128->256 failed; this does not establish a POR result because parent family/targets differ. Only one further POR width candidate is prepared; negative result closes this extension, no automatic width/LR/decoder grid.

Pending: explicitly launch only when channel-line SWv139 is terminal; no automatic queue. Max2GPU with backbone line, normal/80-18/min10/b64/64steps/3e-4/cos80*64, atomic complete last and explicit resumption, temporary keep-awake. Final all144665 best replay/frozen integrity and actual assembly; record training time/launch wall/overlap/peak. Replace PORhead phase, not add to v134: source+POR113+POR140+PERM136+SW104 =5 phases. No other folds or old delivery edits.

Optional: carry measured capacity benefit if positive; otherwise return to architectural/feature mechanisms on retained positive model.

Requested gain analysis completed: [GAIN_ANALYSIS.md](GAIN_ANALYSIS.md), exact frozen fifth144665 predictions, no fit or selection. Point vs local no context gain; logweight+location at64 +0.167638, capacity128 +0.140076, each positive10/16 wells; whole vs v113 +0.328636, nondefault +1.394254/default -0.239516. Source defaultgate and widths frozen, gain conditional density/main continuous cases. Capacity/training differences limit causal claims. V140 explicit POR launch PID88516 after SW terminal, same frozen v113 andzerohead, no v134trainedweights; restorebest and compare against77.869090. Transfer later to each head requires own measured ablation, no automatic sixth phase.

Requested POR gain analysis saved in results/deep_learning/por_joint_density_width_v140_20261001/GAIN_ANALYSIS_ZH.md. Both POR studies actually live, snapshot {"status": "running", "pid": 88516, "stage": "wide256", "epoch": 44, "best": 77.9121028885816, "best_epoch": 30, "bad": 14, "peak_MiB": 924.5458984375}. Partial results only; first amplify measured POR mechanism then assess transfer to PERM/SW. Five phases, no extra folds/delivery changes.

V140 wide256: restored POR=77.912102888582; delta vs v114=+0.364033393689, vs historical=+0.360166086697; actual fifth TOTAL=80.602021989954;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=350.769, peakMiB=924.546.

V140 FINAL confirmed POR77.912102888582, width256 minus128 +0.043012516668; historical+0.360166086697. ActualTOTAL80.602021989954, five source-counted phases. best30/stop48,350.769seconds/924.546MiB;144665 exactreplay/frozen. Keep positivebest/last and new CURRENT_RECIPE; strict80.8 false, no extra folds. Requested amplification verified; next transfer SW conditional readout from original source, no v104 trained weights.

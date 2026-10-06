# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# por_residual_distribution_v145

Focus: latest user explicitly requests512capacity and residualdistribution head mechanism. Residual distribution correction on256head: output48 instead of32, correct16locations+16logweights+16scale raw parameters. Added4112active params vs control, not samecapacity. New scale raw delta0.5*tanh(u/0.5), positive component referencewidth still softplus(raw)+0.025; this is a bounded softplus-parameter residual, not exact multiplicative log-SD. Originalparent rawpluszero retains exactinitialfunction. Proper mixedStudent3/Cauchy NLL andCDF both use corrected width; Cauchy wholemixture still has nofinite mean/variance. Locationdelta bound uses frozen original referencewidth, gate/familyassignments/oldpredictor frozen. No extra mixture components/prediction votes.

Explored/verified: source/trainer/record prepared, untrained. Both from same frozen positivev113 predictor POR77.540454326739; no v140trainedhead loaded. Same ownPORloss/LR3e-4/original9curves206+63amplitude/Q50/train64statistics/fixed fifthseed/scoring/unchangedgrids anddecoding. Source and all historic trained code remain immutable.

Gain: untrained/no score. Compare final actual POR to verified25677.912102888582, then actual combination fixedPERMv136/SWv104. Best selection uses full POR officialscore, include default/nondefault diagnostics. No partial snapshot as finalgain, no historicalf

d all historic trained code remain immutable.

Gain: untrained/no score. Compare final actual POR to verified25677.912102888582, then actual combination fixedPERMv136/SWv104. Best selection uses full POR officialscore, include default/nondefault diagnostics. No partial snapshot as finalgain, no historicalfullOOFclaim.

Not worth continuing: v142 bound2 completed POR77.831487 below bound1 width12877.869090 by0.037603; do not compound that negative bound expansion. SWtransfer v143 currentlyweak; sourcefamily unchanged, no utility/decoder/LRgrid. Earlier linear scaleopening negative does not establish this nonlinear residual result. Only these two predefined userrequested arms.

Pending: explicit launches onlywhen GPUslots free;max2normalGPU/80-18/min10/b64/64steps/cos80*64, admission3GiB, best144665replay/frozenintegrity/actualassembly. Atomicfullstate and explicitresume, process-onlykeepawake, recordwall/overlap/peak. Each replaces PORhead phase: source+POR113+newPORhead+PERM136+SW104=5, not v140plusnewhead sixthphase. StrictoriginalfifthTOTAL>80.8+gain before remainingfolds/fullOOF>82 unmet; prior deliveredpackage immutable.

Optional: after useful gain, transfer mechanism to PERM/SW separately withown controls; noautomatic queue or unbounded sweep.

Explicit userrequested job launch verified by live PID/command; snapshot {"status": "running", "pid": 82756, "stage": "residual256", "epoch": 4, "best": 77.54045432673898, "best_epoch": 0, "bad": 4, "peak_MiB": 923.87060546875}. Supersedes prepared/untrained status. Current control POR25677.912102888582/actualTOTAL80.602021989954, new partial scores not finalgain. Max2GPU,80-18, five total phases; no old trained256head loaded, no folds/deliveryoverwrite.

V145 residual256: restored POR=77.676996071434; delta vs v114=+0.128926576541, vs historical=+0.125059269549; actual fifth TOTAL=80.531489944810;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=312.207, peakMiB=923.871.

V145 residualdistribution finalPOR77.676996071434 vs256delta=-0.23510681714769532, actualTOTAL=80.53148994480973,best17/stop35. This exact boundedrawscale extension negative, no blanketrejection ofallresidualdensity methods. Exact144665/frozenreplay, negativeweights cleanedMiB=22.765783309936523. {"training_seconds": 312.2068451999803, "launch_to_report_wall_seconds": 317.02129, "peak_MiB": 923.87060546875, "overlap_POR512_and192": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. Source/config/results retained, no folds. User384studyv147 takesreleasedslot.



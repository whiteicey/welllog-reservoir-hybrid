# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V357 backbone parallel feature residual

## Focus
Latest user suggests preserve oldbackbonebranch plusnewresidual. Keep original154 expert backbone parameters/enc2 computation, add a separate32width nonlinear17tap featureencoder andzero128projection afterenc2 in each expert. Output h_old+.25*tanh(new(h_old)). Warm original154 readout architecture retained and jointly updated; unlike356 fullyfrozen predictor, readout weights cancoadapt here. Original154only, no trained356/336/349 weights, oneaddedPERMphase (sixdefault/eightretainedothers).

## Explored and verification
Prepared/noGPU/noscore. Each freshbranch128to32/LN/depthwise17/gated64/zero32to128=11104parameters, two22208 plusoriginal82720readout=104928. Original packed output/router/frozenbackbone eval; native166 gradientenabled forward/capture removesdetach so patchgradient traverses frozen downstream operations. New branch masks originalvalid nearest atnativeenc2 resolution; no physicaldepth resampling claim. Local17 versus point sum ofsame stored17kernel, sameparams/init but effectivefunctions differ. Old349rank4enc2+warmreadout negative, new nonlinear localbranch is different; old356 output-onlyresidual best0/14, not newweights initialization.

## Gain
Unproven. Preserve?154initialfunction withzeroexit, ordinaryinitialevaluationexpectednative154, no new numericalpreflight/review. NativeGaussian16 NLL/BCE/RPS1/decoder,

dual_branches.0.norm.bias": 0.0, "residual_branches.0.depth.weight": 3.183403253555298, "residual_branches.0.depth.bias": 0.6948413252830505, "residual_branches.0.expand.weight": 4.552340984344482, "residual_branches.0.expand.bias": 0.8035740852355957, "residual_branches.0.up.weight": 0.0, "residual_branches.0.up.bias": 0.0, "residual_branches.1.down.weight": 3.293905019760132, "residual_branches.1.down.bias": 0.28711530566215515, "residual_branches.1.norm.weight": 5.656854152679443, "residual_branches.1.norm.bias": 0.0, "residual_branches.1.depth.weight": 3.17657732963562, "residual_branches.1.depth.bias": 0.7844030261039734, "residual_branches.1.expand.weight": 4.648634910583496, "residual_branches.1.expand.bias": 0.7975549101829529, "residual_branches.1.up.weight": 0.0, "residual_branches.1.up.bias": 0.0}, "seconds": 111.41831959999399, "peak_MiB": 854.9150390625, "complete_five_folds": false, "early_terminated": true}, "strata": {"default": 97.27870539397452, "continuous": 74.09691194381553}, "current_exit_norms": {"residual_branches.0.up.weight": 1.7500327825546265, "residual_branches.0.up.bias": 0.2155779004096985, "residual_branches.1.up.weight": 1.8259357213974, "residual_branches.1.up.bias": 0.22815413773059845}, "adopted": false, "point_trained": false, "independent_disk_reload": false, "complete_five_folds": false, "remaining_folds_started": false, "patience_control": {"guard_pid": 63112, "training_pid": 46896, "training_live": true, "threshold_target_SOTA": 89.51838939879066, "threshold_basis": "PERM336 with fixedPOR341/SW351, equivalent to retainedassembledTOTAL80.706129723238", "strict": true, "SOTA_broken": false, "effective_patience": 10, "snapshot": {"status": "running", "pid": 46896, "stage": "local", "epoch": 12, "best": 89.40556485691894, "best_epoch": 0, "bad": 12, "peak_MiB": 854.9150390625}, "reason": "Latestuser patience10 untilstrictSOTA, then14; existing epoch-boundary request/bestrestoration", "stop_request_written": true}, "actual_stop_note": "Userpolicy arrived after bad10; guardobserved bad12, existingepoch-boundary stop restoredbest after13, not claim10epoch run", "closed_setting": "native154 enc2 nonlinear32width17tap parallelfeaturebranch pluswarmreadout no gain; no widthkernelLRgrid"}. Bothgradientgroupspositive and currentzeroinitialexitnormsnonzero; restoredzero isbest0 notgradientblock. Latestuserpatience10untilstrictSOTA thenlatch14 recorded; activelegacyguardobservedbad12/stopped13 because instructionarrivedafter10, not10epoch claim. No pointGPU/no matchedgainclaim; existingbest80.706129723238/deliveries preserved.



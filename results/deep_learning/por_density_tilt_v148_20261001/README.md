# v148 proper residual log-density correction

Focus: after192/384/512 capacity and parameter-scale residual failures, preserve useful256location/weight head mechanism and test residual distribution shape directly. Predict256bounded log-density corrections on unchanged original density intervals, no grid/resolution/window/action search.

Explored/verified: implemented base Student3/Cauchy conditionaldensity p(z), correctedlocation/logweights from256head, plus r_i=0.25*tanh(u_i/0.25). Conditionaldensity p_new(z)=p(z)*exp(r_bin(z))/Z, Z=sum_i P_base(interval_i)*exp(r_i). Proper trainingNLL adds logZ-r_bin(y); inference intervalmasses use sameP_base_i*exp(r_i)/Z. Infiniteedge arithmetic safe before CDF, exact0/1 endpoints. Right=True traininglabel bin convention predefined; labels enter likelihood only, not input. Defaultgate/family/sourcecomponent scales frozen. No pseudo-labels or previous prediction blending. Whole Cauchymixture has no finite mean/variance.

Zero newoutputs yield same initial source predictor77.540454326739. Common256hiddeninitialization retained, output288 instead of32 adds65792parameters; total296736active, not samecapacity. Allnewhead weights trainedfromsamefrozenv113, no v140trainedhead loaded. Original9curves206+63amplitude/train64stats/Q50/fifthseed/ownPORlabels/scorer/256intervals/800actions andargmax unchanged.

Gain: untrained/no score. Compare completed realPOR to verified25677.912102888582 and actualfixedPERM136/SW104 assembly. No initiallearning/partialgain claim. True fullPOR selects checkpoint includingdefault/nondefault, no row/wellwise choice. Need all144665bestreplay/frozenintegrity final.

Not worth continuing: width19277.786804/38477.791188/51277.854566 allbelow256, notmonotoniccapacity. v145location/weight/rawscale residual77.676996 negative; do not retain scaleupdate or sweepit. This is a different properlogdensity mechanism, not a delayed decode/reweighting aftertraining.

Pending: explicitoneGPUlaunch after384complete/handleabsent. Normal/max2/admission4GiB/80-18/min10/b64/64steps/3e-4/cos80*64,atomiccomplete last andexplicitresume/process-onlykeepawake. Record realwall/overlap/peak, noOOM. ReplacePORhead phase: source+POR113+POR148+PERM136+SW104=5 phases, no extra stage fromtraining256control. StrictfifthTOTAL>80.8+gain before remainingfolds; fullOOF>82 incomplete andold deliveryimmutable.

Optional: onlyafterusefuldensity-shape evidence transfer to eachoutputhead withown controls. No automatictiltbound/LR/grid sweep.

Verified launch PID74320: proper residual density readout running, snapshot epoch26, partial POR best77.769460078679; not final gain vs256. Initial full-fifth POR77.540454326739 identical to frozen source. Focus distribution readout per latest user direction; no further width sweeps/private backbone launch. Five total source-counted phases; no remaining folds.

V148 tilt: restored POR=77.889012260011; delta vs v114=+0.340942765118, vs historical=+0.337075458126; actual fifth TOTAL=80.595094801383;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=362.568, peakMiB=1312.311.

Final v148: POR77.889012260011, delta vs predefined256 control -0.023090628570; actual TOTAL80.595094801383. Best30/stop48, exact144665/frozen replay. 362.568 training seconds, peak1312.311MiB, single GPU; loading/warmup included in launch-to-report wall (timing.json). Proper interval tilt did not exceed best readout; close this configuration, no further tilt-bound/LR sweep. Negative checkpoints removed with SHA manifest. Best recipe remains v140 TOTAL80.602021989954/POR77.912102888582. Next hypothesis is structured component-shared distribution readout, not yet implemented/launched; READOUT_DIRECTION_ZH.md records evidence. No otherfolds, five phases, fullOOF>82 incomplete.

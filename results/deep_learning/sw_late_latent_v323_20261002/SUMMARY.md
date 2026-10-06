# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V323 SW own-latent late fusion

Focus: keep learnedtask-specific representation close to density/state readout rather than reshape it through three fresh rawMLP branches. Samefresh297encoder/state/FiLM/readout+21600LN384?48GELU?zero48fusion,86865active, original104only. Input104norm128repeat3, central256 of1024, bounded.25tanh, zerooutside. Lateinject atfinalfuse48 BEFORE inheritedstate featurecapture (prependhook), so gate/density see samecorrectedfusedfeatures.

Control: reuse completed321own earlyfusion SW74.166030180492/CONT17.911355427690/best13/earlystop22. Same trainableparams/explicitbody+fusionseeds/Q50/loss/alpha.25ramp5/80-14/min10/b64/64steps/scoring. Newlate omits unused frozen151/154 donorcompute and unusedamplitude63; native104/newrawbranch already206only inboth. This removes nonpredictivework but doesnot claim bitwise identical optimizertrajectory without rerun; compare as historicalsame-capacity control, not freshly simultaneous matched pair. No trained321 weights. Fourauxsamecount but early affects3branches+fused whereaslate onlyfused, so locationcontrast includes gradientpath change, not isolatedreadout causalproof.

Validation/budget: own officialwholeSW best restoration/full144665RAM replay/frozenexact, independentdiskreload onlypromotion/final. Sameoriginal nine206/relative/sourcefold/Q/train-onlypreprocessing/seed. OneSW additionalphase (6standalone/?8a

wnfeaturefusion; v20earlylate inputshape differs from tasktrainednorm128 latent. 315adaptiveprior allocation negative, no gating/alpha sweep. 321cross wholemicroextra lostCONT?.280225; crossnotcarried. OwnCONTgainvs297+.408603/252+.026487 is local singlefold evidence, not robust/fullOOF or purefeature causality.

Results: prepared/untrained, gainunproved. Require usefulCONTandwhole relativeowncontrol, not default-onlymicro. If no benefit closepositiontest, no fusionwidth/bounds/grid. Record realcost reduction onlywhenmeasured, sincehistorical321 includedunuseddonorwork/concurrency/stagger; no isolatedspeedclaim. Timing includeprep/initial/train/bestrestore, pair/concurrentdifferentjobs explicit.

Done: source/trainer/plan prepared andsyntaxcompiled. Next: launchlate when322slotfree, actualfinaldecomposition. Unfinished: gain/808qualification/fullOOF82. Optionalaftergoal: verify donor-removal functionequivalence whenpreparingdelivery, notclaimednow.

PRELAUNCH COMPARISON CLARIFICATION: historical321own manually stopped22 (nominal80/14). Newlate remains80/14; predefine equal-epoch-ceiling22 comparison, save own best/predthrough22 atnormalepoch22 evaluation without extraGPU replay. Any bestafter22 is absolute screen result, not equal-work gain. If earlystopsbefore22, report actualbudget. Unuseddonorcompute removal means no bitwise trajectory claim. Source/trainerchanged onlybeforelaunch, no active/completedfiles modified.

Launched PID37044 at TIMING.json; initialSW74.121752689769, active86865. Training/live gain unproven; historical22epoch ceiling comparison predefined, no bitwise trajectory claim.

V323 late: SW74.137935866983, actualTOTAL80.618139481583; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final conclusion
Latefusion SW74.137935866983/CONT17.618651450885, whole-0.028094313508 and CONT-0.292703976805 versus321own. Predefined bestthrough22 exactly same selectedbest17; historicalcontrol/unusedwork/auxgradientpath limitations remain. Evidence-based boundary stop29/bad12 (request submitted after27), not natural14; bestRAM144665 replay/frozenexact, all fusionparameter deltas/gradient nonzero. Actual6TOTAL80.618139481583; below808/no otherfolds, not adopted. Sourceonly previousown positive retained. Close latefusion position, no width/LR/alpha grid. Individual352.155104s/peak1356.091797MiB, process-to-terminal356.243313s incl preparation/initial/train/bestrestore, partly concurrentPOR; not isolatedspeed comparison. No independentdiskreload.



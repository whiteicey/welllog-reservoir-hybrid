# V322 window-conditioned POR curve-token normalization

Focus: learned amplitude/shape adaptation within positive302 fullcurve-attention encoder. Fresh302 code/init fromoriginal151 only, all newrawbody/curveattention/directreadout+normalizer jointly train onePORphase. No trained302/otherrepair. Realvalid256-window stats versus fixed64train-input reference stats; same162 activezero9?9 shift/scale matrices,76754 totalactive, initialidentity and samepredictionfunction.

Formula: mu=maskedmean(z); shift=.25tanh(A mu); RMS=sqrt(maskedmean((z-shift)^2)+1e-6); scale=exp(.25tanh(B clamp(logRMS,-3,3))). Outputcurve=(rawcurve-shift)/scale, positive scale in[.7788,1.2840]. Reference usesonlytraining inputbank forstats but applies correction to actualcurve; adaptive usesactualvalidwindowstats. Only nine scalar curve tokens normalized beforenumericLinear; original206+63 densebody and151prior untouched. Not outputdenormalization/RevIN reproduction/officialDAIN clone, no gate/loss/decoder added. Mask excludespadding; no labels in stats.

History/novelty: v20already raw206 bypass+63localshape stats, no falseclaim oldshape discarded amplitude; shapeearly relativecontinuous benefit but weakabsolutePOR. adaptive_center_sequence previously learned206 diagonal validwindowmean shift with negativefifth. Currentcrosscurve learnedshift+positiveRMSside-only matchedreference is limitednew intersection, not firstadaptive inputnormalization or promisedhighgain.

Training: original9/features/Q50/fold/seed/scoring unchanged; samefresh302init/alpha.25ramp5/auxCE/RPS/LR3e-4/80-14/min10/b64/64steps,max2GPU. Total6 standalone/?8 assembled andeachPOR+1; noadditionalstages/queue/otherfolds. OwnwholePORbest restored/full144665RAM replay/frozenexact. Independentdiskreload onlyqualification/final. Strictfifth>80.8+gain beforeotherfolds, fullOOF>82 unmet.

Results: prepared/untrained, gainunproven. Firstbridgezero can delay normalizer gradient, later actualupdates recorded; referencefunctionclass differs despiteequalparams. If no usefulwhole/CONTgain vsreference/302 closeboundedconfiguration, no epsilon/range/windowgrid. Timing mustincludeactualwholeattention cost, inputprep/initial/finalrestore and pairwall/concurrency/peak separately.

Done: source/trainer/plan prepared andsyntaxcompiled. Next: launchonlyfreshfree-slot confirmation;321currentlytwoGPU unchanged. Unfinished: actualgain/808qualification/fullOOF82. Optionalaftergoal: independentrepeat if meaningful.

LIVE322 both launched adaptive4512/reference73100; initialPOR77.92638076072386 same both, actualactive76754 (inherited302info76592 is parentcount only; newnorm+162). Allfreshrawbody/curveattention/readout/normalizerjoint. Peaks1815.196/1814.883MiB. {"adaptive": {"status": "running", "pid": 4512, "stage": "adaptive", "epoch": 7, "best": 78.01812646762907, "best_epoch": 6, "bad": 1, "peak_MiB": 1815.19580078125}, "reference": {"status": "running", "pid": 73100, "stage": "reference", "epoch": 2, "best": 77.95497893387893, "best_epoch": 2, "bad": 0, "peak_MiB": 1814.88330078125}}; snapshotnotfinalgain/fullOOF. 321ownretainedoptional8TOTAL80.69780794312307, original5/old8/deliveriesunchanged, strict808false.

LIVE322 updated {"adaptive": {"status": "running", "pid": 4512, "stage": "adaptive", "epoch": 36, "best": 78.02922410702564, "best_epoch": 29, "bad": 7, "peak_MiB": 1815.19580078125}, "reference": {"status": "running", "pid": 73100, "stage": "reference", "epoch": 31, "best": 78.02739942868787, "best_epoch": 29, "bad": 2, "peak_MiB": 1814.88330078125}}; reference caughtup laterbest29, earlier+.007907 pairedsnapshot notfinal/robustgain. ExistingPOR30278.030552794145 retained. V323lateSWfreshownfeaturefusion prepared/untrained, historical321own earlycontrol andpredefined22epochcompare/full80candidate; no GPUlaunch whiletwo322 jobs alive. Optional8TOTAL80.69780794312307 preserved, strict808unmet/nootherfolds/fullOOF82unmet.

V322 final POR78.029224107026, gain+0.102843346302, actualTOTAL80.643328421884; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds1000.973573/peakMiB1815.195801. No otherfolds.

V322 final POR78.027399428688, gain+0.101018667964, actualTOTAL80.642781018383; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds998.817949/peakMiB1814.883301. No otherfolds.

## Final closure
Both naturalstop43/best29. Adaptive/reference whole78.029224107026/78.027399428688, CONT41.671259152479/41.662036621435; pairedwhole+0.001824678338/CONT+0.009222531043, but adaptive below302 whole-0.001328687119 and CONT-0.003820754593. No adoption, micro single-seed mechanism only. Learned normalizer matrix norms and gradients nonzero; full144665 ownbestRAM replay/frozenexact, no independentdiskreload. Pair1058.971775s; individual1000.973573/998.817949s includes preparation/initial/train/restoration, not isolated speed comparison. Only followup is fresh raw+delta token matched324 pair; no bound/window/LR grid or checkpoint chaining.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

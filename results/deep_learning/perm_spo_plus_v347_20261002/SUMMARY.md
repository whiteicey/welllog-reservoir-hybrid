# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM final-mixture SPO+ v347

Focus: preserve original154prior plus fresh336rational/cross raw encoder/readout code and alloriginal losses; add one decision-loss term, no newparameters or decoder changes. Allnewbody/readout/rational/cross jointlytrain onePERMphase/max8, onlyoriginal154 checkpoint, no336/346repairweights.

Hypothesis/control: same37386params/init freshreference versus fixedlambda.1 SPO+. s=P@existingEvaluator.utilities for finaljoint[g,(1-g)*mixedmass], notauxcomponent; actualownphysicaltrainingtruth y supplies u=max(0,1-|log10(action)-log10(clamp(y,1e-5))|). DEFAULTatom.01 remains, butdo notreplace trueDEFAULTlabels by.01. a*=argmax(training u); loss=max(2s-u)-2s[a*]+u[a*], allvalid/Qmean/3, originalloss unchanged. Traininga* is usualcostsupervision, never predictorinput orvalidationoracle. OriginalEvaluator actions/K/copiedbuffers, sameexactdecoder; chunk1024, maskcoremax8192, no enlargedpopulation. Reference computes sameextra term butcoefficient0 for fair computation/RNG and recordsdiagnostics; no newinferenceparameters.

Explored/verification: primarySPO+paper/authorcode andcurrentkernel/mass/label semantics checked in ../cross_domain_research_20261001/PERM_SPO_PLUS_MIXTURE_RESEARCH_20261002.md. Currentdata actualtruth CPU only traininganchor owncolumn is transferred into NaNoutside-training GPUbuffer; lossfinite check catches anynontrainaccess. Proper/CE

reference explicitlysolo launchedPID70228 after346pairterminal/globalGPU0; spo armnotlaunched/noautoqueue. Zero-coeffreferencecomputesSPO diagnostics butno addedloss, samefresh33637386; no oldtrainedrepairweights. In modeldiskreload construct/configure originaldecision buffersbeforeloadingstate, originaldecoder unchanged. Gainunproven.

V347 final PERM89.518389398791, gain+0.112824541872, actualTOTAL80.651964007649; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds169.793728/peakMiB907.652832. No otherfolds.

Reference natural19/best5 PERM89.518389398791/CONT74.009637083197/DEFAULT97.494439755942; original336 predictions exactlysame=True, changedrows0; no baseline alteration.169.793729s/907.653MiBsolo inclprep/traintruthtransfer/warmup/restoration. ReferencecomputedSPO0coefficient extra graph notnewloss.144665RAMbest replay/frozenexact notdiskreload/fullOOF. SPOexplicitPID58256afterreferenceterminal/globalGPU0, same37386params/init andK/actions, gainunproven/pairedpending. Current341TOTAL80.704705566183 remains.

V347 final PERM89.518347404301, gain+0.112782547382, actualTOTAL80.651949309578; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds171.090512/peakMiB907.652832. No otherfolds.

## Terminal paired conclusion
Bothnatural19/best5, referencePERM89.518389398791/CONT74.009637083197/DEFAULT97.494439755942 exactlyreproduceshistorical336predictions, versusSPO89.518347404301/CONT74.008876185089/DEFAULT97.494767488904. SPO-reference whole-.000041994490/CONT-.000760898108/DEFAULT+.000327732962 => no whole+CONTgain/no adoption, close fixedlambda.1SPO+ setting/no lambda-temperature-decoder-LRgrid, not all decisionlossinvalid. Native154CONT still-.088035758727. Same37386params/init/K/actions800/fullprobabilitysupport257, actualowntrainingtruth578891rows/heldouttruth NaN, no validationoracle/decoderchange, allnewencoderreadoutjointonePERMphase/max8/no repairweights. Individual169.793728/171.090512s solo includestraintruthtransfer/prep/warmup/restoration, pair398.121619s includes explicitserialgap/stagger221.925, peaks907.653MiBboth.144665RAMbestreplay/frozenexact notmodeldiskreload/single seed,19prefixsamework. Current341TOTAL80.704705566183 remains/strict808fullOOF82unmet/nootherfolds. After terminal, auxiliary documentation snapshot writer initially expected epoch field andfailedKeyError; no fileswritten/noGPUfailure/no restart, corrected by reading terminalreport. Nativeexpert earlyLoRA research inprogress, no source/GPUyet.



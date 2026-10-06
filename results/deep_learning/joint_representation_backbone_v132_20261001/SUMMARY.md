# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v132 backbone route: train-only hidden representation alignment
Focus: overcome weak randomly trained newbackbone representation without retaining old predictor at inference. Student is identical random v130hierarchicalarchitecture/initialpredictioncapacity937515, now trained on original3channel labels plus0.1normalized hiddenfeatureMSE from originalown-fold teacher. Teacher2x128hiddenvectors, not targetpredictions; no pseudo-label or predictiondistillation. Only64trainingwell windows feedteacher.
Explored/verified: implemented frozenown-foldteacher behind training-only branch and2new96->128auxprojections24768params. Teacher originalsource alreadycounted forQ, thus standalone2trainingphases still; no oldtrainedweights copiedinto student. Student supervisedNLL/BCE unchanged and original3outputTOTAL bestselection/CDF/argmax. Heldoutvalidation/inference nevercallteacher/auxloss; auxprojectionsnotusedinprediction. Teacherfrozen signature checked, own source original64train provenance. Architecture/init/labels unchangedvs v130 butadditionalauxobjective/trainingparams, not equalactivecapacity. No teacher predictions used in loss/inference, no extra labels/data/pretrainedexternalweights. Ignore discarded modelreturn predictions; only prehookhiddenfeatures read.
Gain: untrained, unknown. Compare actual3targetfifthTOTAL to v13078.955352487050, retained compact80.493464316205 and oldfu

tudent is identical random v130hierarchicalarchitecture/initialpredictioncapacity937515, now trained on original3channel labels plus0.1normalized hiddenfeatureMSE from originalown-fold teacher. Teacher2x128hiddenvectors, not targetpredictions; no pseudo-label or predictiondistillation. Only64trainingwell windows feedteacher.
Explored/verified: implemented frozenown-foldteacher behind training-only branch and2new96->128auxprojections24768params. Teacher originalsource alreadycounted forQ, thus standalone2trainingphases still; no oldtrainedweights copiedinto student. Student supervisedNLL/BCE unchanged and original3outputTOTAL bestselection/CDF/argmax. Heldoutvalidation/inference nevercallteacher/auxloss; auxprojectionsnotusedinprediction. Teacherfrozen signature checked, own source original64train provenance. Architecture/init/labels unchangedvs v130 butadditionalauxobjective/trainingparams, not equalactivecapacity. No teacher predictions used in loss/inference, no extra labels/data/pretrainedexternalweights. Ignore discarded modelreturn predictions; only prehookhiddenfeatures read.
Gain: untrained, unknown. Compare actual3targetfifthTOTAL to v13078.955352487050, retained compact80.493464316205 and oldfullCV-fifth80.504197424477. Auxpositive evenbelowSOTA canjustify subsequent isolatedstudy, no initiallearninggainclaim. FullOOF>82 unresolved.
Not worth continuing: v127privateConvGRU/v130purelabelnewbackbone closed; this changes trainingrepresentation rather than repeating purelabelwidth/LR sweeps. No alignmentcoefficientgrid thisbatch without measuredgain.
Pending: explicit onebackbonejob beside v131PORhead, max2/normalGPU.512context/original206+relative/input-onlybank/Q50/fixedsplit/seed/80-18/min10/b64/64steps/LR6e-4/cos80*64. Completeatomicstate/explicitresume/epochstop, best144665x3 replay+frozenteacher integrity. Two dependencyphases originalsource/newstudent include teacherpretrainingcost; potentialprivatePOR+fixedPERM/SWrecipe5. StrictTOTAL>80.8+gain before remainingfolds. Temporarykeepawake, record additionalteachercompute/auxmemory actual wall; preserveoldsource/deliveries.
Optional: if alignment restores meaningfulrepresentation, continue a new privatechannel study from selectedstudent with originalprotocol; no outputvoting or roworacle.

V132 failed before anyoptimizerstep: fixedteacher256window cannot read512. Preserve failedsource/config, cleanupinitialstate; new v132r1 trial keepsstudent512, teachercentral256/alignedcommon128core, aux24832actualparams. No measuredgain/negative exists for failedv132; r1 explicitlaunch next, max2 withPORv131.



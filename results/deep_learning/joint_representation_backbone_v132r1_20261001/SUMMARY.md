# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v132r1 backbone route: train-only hidden representation alignment
Focus: overcome weak randomly trained newbackbone representation without retaining old predictor at inference. Student is identical random v130hierarchicalarchitecture/initialpredictioncapacity937515, now trained on original3channel labels plus0.1normalized hiddenfeatureMSE from originalown-fold teacher. Teacher2x128hiddenvectors, not targetpredictions; no pseudo-label or predictiondistillation. Only64trainingwell windows feedteacher.
Explored/verified: implemented frozenown-foldteacher behind training-only branch and2new96->128auxprojections24832params. Teacher originalsource alreadycounted forQ, thus standalone2trainingphases still; no oldtrainedweights copiedinto student. Student supervisedNLL/BCE unchanged and original3outputTOTAL bestselection/CDF/argmax. Heldoutvalidation/inference nevercallteacher/auxloss; auxprojectionsnotusedinprediction. Teacherfrozen signature checked, own source original64train provenance. Architecture/init/labels unchangedvs v130 butadditionalauxobjective/trainingparams, not equalactivecapacity. No teacher predictions used in loss/inference, no extra labels/data/pretrainedexternalweights. Ignore discarded modelreturn predictions; only prehookhiddenfeatures read.
Gain: untrained, unknown. Compare actual3targetfifthTOTAL to v13078.955352487050, retained compact80.493464316205 and old

teatomicstate/explicitresume/epochstop, best144665x3 replay+frozenteacher integrity. Two dependencyphases originalsource/newstudent include teacherpretrainingcost; potentialprivatePOR+fixedPERM/SWrecipe5. StrictTOTAL>80.8+gain before remainingfolds. Temporarykeepawake, record additionalteachercompute/auxmemory actual wall; preserveoldsource/deliveries.
Optional: if alignment restores meaningfulrepresentation, continue a new privatechannel study from selectedstudent with originalprotocol; no outputvoting or roworacle.

Repair/provenance: v132 failed before firstoptimizerstep because teacher512window violated fixedmaximumtokens; not a measurednegative result. r1 is newsource/config/trial, same randomstudent and24832auxparams. Student reads512, teacher readsnativecentral256; align exactcommoncentral128 supervisedmask, owntrainingwells only. No positional limit edits or trainedteacherhash changes. Same labels/score/Q/budget/objective0.1; alignment now well-defined. Preserved failedsource/config/trace, deleted unused initialstate with manifest.

Explicitlaunch verified by actualcommand PID24968; snapshot {"status": "running", "pid": 24968, "stage": "aligned", "epoch": 7, "best": 77.89858094587699, "best_epoch": 6, "bad": 1, "peak_MiB": 1121.14794921875}. Trainingonly, no finalgain;max2 acrossroutes. POR historical+0.25 and originalfullOOF>82 unmet, no otherfolds/delivery changes.

Explicit actual process verified PID24968, mode aligned, runtime snapshot {"status": "running", "pid": 24968, "stage": "aligned", "epoch": 22, "best": 78.90513703645937, "best_epoch": 18, "bad": 4, "peak_MiB": 1121.14794921875}. Trainingonly, no finalgain;2sharedslots. CurrentverifiedPOR77.729014112744/TOTAL80.543755697579,5phases; strictPOR+0.25 andTOTAL80.8/fullOOF82 unmet. No otherfolds or delivery changes.

V132r1 joint source verified: actual original-fifth scores{'POR': 74.69914236435433, 'PERM': 89.11417805271635, 'SW': 73.25866038340264, 'TOTAL': 79.24023616194795},2 stages incl originalsourceQ; strict808=False; no remainingfolds. best/stop=37/55, seconds=718.312, peakMiB=1121.148;144665x3 replay exact.

V132r1 finalpositive alignment: originalfifthTOTAL79.240236161948 best37/stop55; gainvsnoaux {"POR": 0.9686305621260374, "PERM": 0.4627241474484407, "SW": -0.47902555813300296, "TOTAL": 0.2848836748982251}. Predictionarchitecture identical, extra24832trainingauxparams; not equalactivecapacity. All144665x3replay/frozenteacherexact. Keepbest/last aspositivecontrol despitebelowSOTA/gate, no otherfolds. Newv135 replacesinputstemwithcurve tokens whilekeepingalignment, untrained.



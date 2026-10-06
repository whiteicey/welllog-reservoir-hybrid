# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v257 small continuous-feature backbone branch and mass readout

## Focus
Carry254 raw original206+relative feature mechanism (+0.000987 versus repeated-hidden samecapacity, bounded17epochs) into small nonlinear crosscurve/depth encoder. Stem207->32GELU, two32channel LayerNorm/depthwise5/FFN64 residual blocks with dilations1/4, concatenate frozen154128hidden with new32hidden then160->32->257 mass/background readout. Point versus local both29185storedparameters/sameweights/init/loss; point collapses each5tap kernel bysum, local keeps actual temporalcontext. Effective degrees of temporal freedom differ, not identical functional expressivity.

## Explored and validation
Prepared/held while256live. Full154/gatefrozen, no trained254/248weights. Architecturebranch and readout trained jointly in ONE added PERM phase; backbone extension retains original predictor. Original9curves/206derivedfeatures, train-onlystats/Q50/fold5/256bins/800actions, original CE-normalized continuousNLL+jointRPS1/AdamW1e-4/noSAM/80-14. Newloss256 not compounded until proved. Samezero finalprojection/.001background, originalinitialscore separatelyreported. Final RAMrestore/frozen/all144665actualcomposition, not diskreload claim.

## Gain
Unproved. Local-minus-point matchedstoredcapacity and absolute retained248PERM89.413465595388. New29185=stem6656+two4448blocks+head13633. Bothraw branches; this isolates tem

 proved. Samezero finalprojection/.001background, originalinitialscore separatelyreported. Final RAMrestore/frozen/all144665actualcomposition, not diskreload claim.

## Gain
Unproved. Local-minus-point matchedstoredcapacity and absolute retained248PERM89.413465595388. New29185=stem6656+two4448blocks+head13633. Bothraw branches; this isolates temporal communication given nonlinear crosscurve encoding, not raw-versus-hidden feature source. Sixalone/sevenwith retainedSW252/total<=8, no sequentialrepairchain.

## Not worth continuing
254 raw tiny local benefit allows isolatedcarry, absolutebelowretained does not prove generalnewfeature benefit. 239 weak depthwisehidden/readout did nothelp; here trains new crosscurve nonlinear feature branch. If both fail, no dilation/depth/width grid. Do not restart failed wholebody/SAM scope184.

## Timing
Record individual and actual concurrent processcreation-to-finalreport wallclock/peakMiB; startup/model/data/initialevalincluded, codeprep/CPUassemblyexcluded; no serialspeedup claim.

257 actuallaunch pointPID64632/localPID60836, bothsameinitial/29185storedparams. Epoch3best point89.406580519480 versuslocal89.406499698385 (best2both); early/unproved. Bothgradienttrain/noerrors/OOM, peaks790.941MiB,10.2-10.6s recentepoch/concurrentpair (not serialbenchmark). Original154frozen/onePERMaddition; no otherfolds.

V257 final PERM89.409216200327, gain+0.003651343409, actualTOTAL80.613753388187; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/17, trainingseconds194.156798/peakMiB790.940918. No otherfolds.

V257 final PERM89.409182616679, gain+0.003617759760, actualTOTAL80.613741633910; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/17, trainingseconds194.175286/peakMiB790.940918. No otherfolds.

257 bounded17both point89.409216200327 best5/local89.409182616679 best6, local-minus-point -0.000033583648, bothbelowretained24889.413465595388. No temporal mechanism benefit here; no branch depth/dilation/width grid, no fullconvergence/generaltemporalnegative claim. RAMbest/frozen/all144665actualcomposition done; preserve source/config/score/SHA and remove redundant checkpoints forboth unpromisingbranches withmanifest. Currentbest248/252 unchanged. Pairwall201.105s versusindividual194.157/194.175s/790.941MiB, startup/data/model/initialevalincluded/prepCPUassemblyexcluded, no measuredserialspeedup.258reference/wellinputcontext launched onlyafter257PIDs terminal,max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337unchanged.



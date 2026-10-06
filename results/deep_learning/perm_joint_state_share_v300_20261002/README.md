# PERM continuous/state coupling v300

## Focus / evidence
V298 conditional allocation produced continuous74.091162808666 vs scalar74.037494684582 (+.053668) and retained29174.001066339761 (+.090096), but default97.287981640665 vs29197.494555208792 lost, wholePERM89.409738 below29189.515555. Gate network frozen did not fix official action because utility combines state probability and changed continuous mass. Test whether jointly adapting state representation can retain the continuous signal while recovering this loss.

Fresh new291 sparsebody/readout and298conditional allocation code only, ORIGINAL154 checkpoint frozen only, no trained291/298repair weights. Add zero48->1 state residual2*tanh(raw/2) from same newfeatures. Fixed control registers statehead but freezes it0; joint arm trains49 additional parameters; samebody/readout/share/initialtensors/loss/budget and decoder. Active36515vs36564 difference49 disclosed as intentional update-scope comparison, not exact active-capacity ablation. Allnew body/readout/share joint3e-4 in both, optionalstate3e-4 in joint. Same own trainingBCE/Q50/equal-trainwell continuousNLL/jointRPS1/directCE1; no changed defaultthreshold/score/actions or posthoc perrow winner. Newstate gradients also shape newrepresentation.

## Explored and verification
Prepared/untrained. Ordinary initial154 replay, actual encoder/readout/share/state gradients, original154 frozen checks, atomic current/best/optimizer/RNG. Own wholePERM-best, then actual144665 assembly/default-continuous diagnostics; no stratum-specific checkpoint selection. Fixed arm should replay298conditional trajectory if deterministic same inputs, but requires actual evidence; do not assert equivalence from code alone.

## Gain
Unproved. Primary joint-minus-fixed wholePERM and stratumtradeoff; absolute retained29189.5155547932572/continuous74.001066339761 and v298continuous74.091162808666. Preserve oldmodels, only adopt complete channel candidate after actualcomposition. ONE PERMaddedphase from154, max8whole, not alpha-trained then gate-trained phases. Strictfifth>80.8/usefulgain before otherfolds; fullOOF>82 unmet.

## Not worth following
If joint state fails to recover overall score or destroys continuousgain, close this directcoupling setting rather than sweep statebounds/LR or trainingweights. Default-state and continuous objectives may interfere in the samefeatures; positive continuous diagnostics are mechanism evidence, not deliverable improvement. Prior input may still lack discriminative information. No claim all adaptive allocation fails.

## Timing
Record individual and paired totalwall incl data/initial/train/finalRAMrestore; codeprep/CPUassemblyexcluded, actual serial/overlap/gap and peakMiB, max2GPU. No redundant preflight/review.

V300 final PERM89.409738092787, gain+0.004173235869, actualTOTAL80.613936050548; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds210.240520/peakMiB787.851074. No otherfolds.

V300 final PERM89.419650223819, gain+0.014085366900, actualTOTAL80.617405296409; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds438.768973/peakMiB787.854004. No otherfolds.


## Final paired outcome
Fixed PERM89.409738092787, joint89.419650223819, pairedwhole+0.009912131032/pairedcontinuous+0.412532501309, joint-minus291-0.095904569438. Strata recorded separately; no adoption or statebound/LR grid. Fixed control exactlyreplays298conditional144665preds=True; ownbestRAM/frozen/actualassembly, no independentdiskreload. Active49parameterdifference intendedupdate-scope. Individual joint438.769s/fixed210.241s, pairwall447.835s partialoverlap/no speedranking. Localpositivecontrols retained; checkpointcleanup for unpromising redundantstates remains policyblocked, no deletion claim. Currentoptional8/default5 intact, strict808/OOF82 unmet.
Next bounded diagnosis (prepared, NOT evaluated): v300 joint-best29 fixedcheckpoint, zero only newstate delta at inference and retain its jointlytrained encoder/density/share; compare to existingfull89.419650/continuous74.503695 and291whole89.515555. Onevariant/no weightsweep/no roworacle/no extra phase, same originalpopulation/scoring. This separates inference-state delta from the representation benefit of state-supervised jointtraining; no claimthat pruned variant was checkpointselected or independently trained. Evaluate only after an actualGPUslot frees, no trainingstop. If no gain, close inference-state pruning, then consider ONE new same-phase decisionutility migration (oldv87/v171 negatives disclosed), not arbitrary calibrationgrid.


## Fixed-checkpoint inference state pruning
Actual PERM89.406477687966, versusfull300-0.013172535853/retained291-0.109077105291; default97.280204083215/continuous74.096685623378. Actualoptional8 TOTAL80.655335735422, strict808=False; onlyzero newstate at fixedjointbest29, allotherweights unchanged, source/diskcheckpoint hashes preserved. ONE predefined ablation, no stateweight sweep/roworacle/newphase, not optimized for prunedscore. Do not equatepruninganalysis with independent remainingfold verification. GPUeval3.011s incldata/init/diskload/eval/CPUcomposition/record, peak786.465MiB; not trainingtiming.

Fixed best29 state pruning completed: PERM89.406477687966 versus unpruned89.419650223819 (-.013172535853), continuous74.096685623378 versus74.503695309976 (-.407009686598). Non-state weights unchanged, one variant/no retraining/no per-row selection. Close inference pruning; original291 remains selected. GPU evaluation3.010677s/peak786.465MiB includes data/init/load/eval/assembly, not training timing.

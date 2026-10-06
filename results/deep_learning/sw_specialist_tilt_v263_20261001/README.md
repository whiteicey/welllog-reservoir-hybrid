# V263 new specialist representation with direct prior-relative readout

## Focus
V262 training-input-only CPUfloat32 16-anchor diagnostic found continuous newexpert mixing alpha mean.000228 TCN(epoch9)/.000435GRU(epoch12), from initial.01. Newexpert trainCE decreases but finalprediction nearly oldprior. This establishes collapse on sampled trainingwindows only, not allvalidation/allwells. Test readout with no learnable mixture gate.

## Explored and validation
Prepared/untrained. Same new207input TCN64/dilations1,4,16,64 versusBiGRU64, original104 frozen. Replace257 expertmass/alpha output with256 prior-relative mass logits. p=softmax(log(.999*oldprior+.001/256)+3*tanh(newlogits/3)), residualzeroinitial. No learnedgate can ignore newencoder; independent ownlabel residualsoftmaxCE.5 still trains newfeatures. Final conditionalCE andownCE continuousQ-normalized, halfhard-halftolerance targets, no teacher or validation fitting. Encoder/readout jointly3e-4/80-14/min10/SAM0/64x64. No262 repairedcheckpoint; fresh104initialization. Fixed.001floor vs262.01uniformmix initialfunction differs and separately reported. Betweenarchitectures capacity differs, no samecapacity attribution. OneSWaddition/sixalone/sevenwith retained248 <=8.

## Gain
Unproved, strict808/fullOOF82 unmet. Absolute retainedSW25274.149228207812. Need learned-vsinitial and finalscore on all144665; currentpositive/default5/delivery unchanged. No standalone expert prediction voting. Existing density is retained within one model's residual distribution.

## Not worth continuing
Do not keep repeated sigmoid-mixture trials once independent newfeatures are suppressed. If direct priorrelative newfeatures still cannot improve, close this specialist/residual combination rather than width/bound grid. No claim all recurrent/TCN models fail. Knownpadding/float32recurrent differences unchanged; previous low alpha is CPUtraining snapshot, not exactGPU replay.

## Timing
Record individual/concurrentwall/peak. Include startup/data/initialeval/finalrestore; exclude preparation/CPUdiagnostic. Max2GPU, no default serialspeedup claim.

V263 liveverified PIDs9332/45272. TCNepoch2 SW74.083129319825; GRUepoch4 SW74.071211629977; both bestinitial74.121701580675, no learnedgain yet. Finite losses/gradients, peaks937.782/1295.659MiB. Continue80-14 max2. V262 negative checkpoints removed after full144665composition; CHECKPOINT_DELETION recordsSHA/paths; original104/retained252/delivery untouched. Atlas1023experiments/194deadends/216unresolved, no completeancestry claim.

V263 tcn final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V263 bigru final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V263 FINAL bothSW74.121701580675/best0/natural14stop/no learnedgain, actualTOTAL80.612457481375, full144665composition/RAMbest/frozenverified/no diskreload. TCN188.605411s/937.782MiB, GRU151.468424s/1295.659MiB, concurrentwall194.192s inclstartup/data/initialeval/finalrestore/excludesprep. Close thisnewrawspecialist/readout route, no width/bound scan. V264 launched matched12609PERM wellmean/softworstrisk, fresh154 only/no repairedweights/onePERMaddition/max2/80-14. Gainsunproved. Research nextPORlateblock V-REx documented only, not trained; docs/deep_learning/research/representation_large_gain_pivot_20261001.md.

Finished unpromising best/last checkpoints deleted after fullcomposition; deletionmanifest preservesSHA/paths/reasons. Sources/logs/compactscore evidence kept. Currentpositives/active264 preserved.

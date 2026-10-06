# PERM bounded adaptive component share v298

## Focus / predefined hypothesis
The positive291 sparse raw encoder jointlylearns a direct continuous density but mixes it at fixed .3 even where it is less/more reliable than original154 prior. Learn bounded component share jointly with this fresh encoder/readout: alpha=nominal_ramp*(1+tanh(raw/.3)), nominal0->.3 over5epochs, actual[0,.6]. Original154 gate/prior frozen, no291repair weights. No architecture/backbone replacement of protected prior, no teacher or validation-guided perrow selection.

Scalar control learns one bias (48 zero share weights frozen); conditional arm learns the same zero-init48->1 module. Both same registered parameters and original291 random initialization/inference/protocol, active36467 versus36515: disclose48parameter difference, not exact active-capacity ablation. Allnew sparse body/readout plus active share parameters jointly3e-4. DirectcomponentCE1 remains independent ofalpha to prevent branch starvation; mixedNLL/equal-trainingwellrisk/jointRPS1 learns reliability through training labels only. Both arms pointselector only, oldPERM default gate frozen, hence objective targets continuous density. Same original206+relative9curves/Q50/originalfold5/seed20261010/80-14/min10/b64/64steps.

## Explored and verification
Prepared, untrained. Ordinary initial evaluation must replay154 within1e-3, actual newgroup gradients recorded; no additional preflight. Atomic laststate includes actual learnedshare weights, nominal schedule buffer, optimizers/RNG and own-best state. Nominalalpha .3 is NOT actual learnedalpha; training min/mean/max logged from actualvalid inputs. OfficialPERM-best selection and actual144665 assembly after completion, RAM replay is not independent diskreload.

## Gain
Unproved. Primary conditional-minus-scalar finalPERM; absolute retained29189.5155547932572 and default15489.40556485691894. One addedPERMphase from original154, not chained onto trained291; whole method<=8. No strict80.8 pass/fullOOF82 claim, no remainingfolds. Earlier adaptive density/background routers existed; this is new directcomponent reliability migration, not first adaptive mixture.

## Not worth continuing
If conditional fails scalar/retained and continuous metric does not improve, close this bounded share setting, no boundary/temperature/LR grid. Shortcut feature-dependent allocation can fit trainwells; broad conditional densities may deserve averaging instead of pointwise confidence. A useful scalar gain alone does not prove conditional mechanism.

## Timing
Record individual data/initial/train/finalRAMreplay seconds/peakmemory and same-work paired wall including staggered overlap/gaps; exclude codeprep and CPU assembly. Max2 is a limit, not automatically a speedup.

V298 final PERM89.466090565545, gain+0.060525708626, actualTOTAL80.633659416013; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds214.688933/peakMiB788.700195. No otherfolds.

V298 final PERM89.409738092787, gain+0.004173235869, actualTOTAL80.613936050548; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds194.641268/peakMiB788.701660. No otherfolds.


## Final paired negative
Scalar PERM89.466090565545, conditional89.409738092787; conditional-minus-scalar-0.056352472757, scalar-minus291-0.049464227712. Continuous scalar74.037494684582/conditional74.091162808666 versus29174.001066339761. Actual selected scalarshare0.184601694345, nominal .3 is not actual alpha. Newbody/readout/allocation grads nonzero, original154 frozen/RAMbest replay/144665actualassembly. No adoption: close this early adaptive allocation setting, retain original291 and earlier candidates. Pairwall279.053s, individual scalar214.689s/conditional194.641s; partial overlap, no speedclaim. Redundant negative weights eligible for manifest cleanup after this record; compact predictions/source/config evidence retained.
Clarification of v298: conditional continuous74.091162808666 vs scalar74.037494684582 (paired+.053668124085) and retained29174.001066339761 (+.090096468906), while default97.287981640665 versus29197.494555208792 loses. Frozen gate does NOT freeze official default action since density changes utility competition. No adoption, but retain conditional positive-mechanism/control weights; future state+density coupling is a distinct fresh joint phase, not adaptive bound grid or stacked repair.

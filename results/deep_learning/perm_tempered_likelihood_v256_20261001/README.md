# v256 tempered continuous likelihood on retained PERM readout

## Focus
Same248 mass/background12609param architecture/init/seed/normalizedcontinuousNLL/jointRPS1; change own conditional onehot bin loss from -log p(y) to (1-p(y)^q)/q with fixed q0.2. Numerically -expm1(q*logp)/q. CE control q->0 uses original -logp. Low probability rows receive smaller gradient p^q; no sample deletion, quality reclassification, target changes, validation fitting or pseudolabels. This is an objective change, not a claim proper loglikelihood/calibration preserved.

## Explored and validation
Prepared/held while255live. Original154/gate fullyfrozen, same continuous Q-normalization and original joint255edgeRPS1/BCE. Training labels only/no threshold tuning. Both12609params/LR1e-4/80-14/noSAM/64steps. Initialsame.001uniformbackground measured separately. Final RAMrestore/frozen/full144665composition required; no diskreload claim.

## Gain
Unproved. Predefined q02-minus-newCE control and absolute vs248PERM89.413465595388. OnejointPERMaddition/sixalone/sevenwith retainedSW252, total<=8. No trained248/255weights loaded. Does not infer difficult rows are anomalous; tests whether emphasis on extremely lowprobabilities harms this constrained predictor.

## Not worth continuing
Previous shape/context/control changes gave no material absolute gain; preserve effective prior/mass/RPS mechanisms. Fixed q0.2 only, no q/coefficient grid if negative. The gradient emphasis hypothesis is unproven, labelnoise/irreducibleerror not claimed.

## Timing
Individual and concurrent processcreation-to-finalreportwallclock, peakMiB; startup/model/data/initialevalincluded, codeprep/CPUassemblyexcluded; no serialspeedup claim.

256 actual launch CE PID62520/q02 PID62420, sameinitial89.404398334646/12609params. Epoch1CE89.406764227678/q0289.406750098120, no earlyrelative gain, peak789.904MiB/noerrors/OOM. Lossmagnitudes differ byobjective/not accuracy evidence. Bothrealgradient updates/max2/original154 frozen/onePERMaddition. No otherfolds.

V256 final PERM89.413465595388, gain+0.007900738469, actualTOTAL80.615240676458; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/17, trainingseconds180.099758/peakMiB789.903809. No otherfolds.

V256 final PERM89.413048870990, gain+0.007484014071, actualTOTAL80.615094822919; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/17, trainingseconds180.489797/peakMiB789.903809. No otherfolds.

256 bounded17both CE89.413465595388 best10 vsq0289.413048870990 best6, delta-0.000416724398. No loss-mechanism gain; no qgrid. RestoredRAM/frozen/full144665actualassembly; CE checkpoint state/predictions exact existing248, verified CPU tensor comparison. Remove redundantCE best/last and unpromisingq02 checkpoints withmanifest after this evidence; preserve248positive weights. Pairwall186.983s, individuals180.100/180.490s/789.904MiB, startup/data/model/initialevalincluded/prepCPUassemblyexcluded,no measuredserialspeedup.257newsmallfeaturebackbonepoint/local launched onlyafter256PIDs terminal/max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337unchanged.

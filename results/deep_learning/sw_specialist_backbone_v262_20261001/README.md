# V262 new continuous-value specialist backbone

## Focus
User requests a direction with larger potential gains instead of tiny readout repairs. Build a new private continuous-value encoder directly from original206features+relative, trained with density readout in one phase. Retain frozen104 prior and atom branch. Hypothesis: new target-specific features can recover continuous events that frozen hidden corrections cannot. Gain NOT guaranteed.

## Explored and validation
Prepared, not launched. Two architecture candidates: width64 residual depthwiseTCN four blocks/dilations1,4,16,64/receptive341 versus bidirectionalGRU64/full1024window. Parameters differ; not a same-capacity mechanism ablation. Existing source104/input-only bank/Q/scoring unchanged. Neither uses old hidden features or teacher labels. New expert256-bin logits and learned sigmoid mixing alpha initial.01; initialexpert uniform/nonidentity explicitly scored. Gaussian prior and atom gate frozen. Own real continuous-label halfhard/halftoleranceCE coefficient.5 gives expert gradients independently of initialsmallalpha; mixtureCE and ownCE separately normalized over continuousQ, BCEconstant, all/3. No RPS/RNC/SAM; newencoder/readout3e-4 joint80-14/64x64. No247/250/252 repair checkpoints, one newSWphase/sixalone/sevenwith retainedPERM248 <=8.

## Gain
Unproved. Absolute retainedSW25274.149228207812/currentoptionalTOTAL80.6248570593371; distinguish learned-versusinitial gains and newarchitecture comparisons with different capacity. Initialuniformmix score changes not learnedgain. Actualfull144665/RAMbest/frozen then diskreload if qualification; no fullOOF claim.

## Not worth continuing
246 additivehead and247 auxiliary read frozenhidden and had negligible gains; current branch instead learns newrawinput representation with independent own-label supervision. This is not ordinary width sweep. Stop if newexpert learns trainingCE but official score remains far below controls with no trajectory; no automatic biggernetwork sweep. Does not establish labelnoise or inherent tasklimit.

## Timing
Record individual and actualconcurrent wall/peak, startup/data/initialeval/finalrestore included, preparation/CPUassembly excluded. CPU-only recurrent source here, CUDAfloat32GRU in runtime, sequential windows not extraGPUjob. Padding zero masks; GRU can see maskedpadding lengths, no labels. No serialspeedup claim.

V261 FINAL (recorder copied label V259 is historical typo; this evidence is V261): CE74.140580389878/RNC74.134371736808, delta-0.006208653070, bothearly14/best7and3; actualTOTAL80.619065064596/80.616892036022, no808. Individual209.635357/209.040677s, peak804.753MiB, pairwall217.325s. CEpredictionequal250=True, tensorstateidentity notchecked. Close bounded rank.1; no convergence/generalRNC claim. V262 launched two new continuousspecialist raw207 backbones TCN98497/BiGRU96513 parameters, matched initialSW74.122050668856/nonidentityalpha.01. OneSWphase; owncomponentCE.5+normalizedmixtureCE jointly trained, no teacher/no oldrepair weights. Gainunproved, preserve currentbest/default5/delivery, nootherfolds.

V262 live verified PIDs69224/44720. TCN epoch3/bestinitial74.122050668856, BiGRUepoch4/best2 74.122942685760. No meaningful gain yet. Trainable98497/96513, peak938.692/1295.686MiB respectively, finite owncomponentCE and losses. Continue80/14 max2; no qualification/otherfolds. Prior and original atom frozen. V261 checkpoints removed after fullcomposition; CEpred equal retained250 control; deletionmanifest kept. Atlas1020experiments/193deadends/215unresolved, no completeancestry claim.

V262 train-only16anchor CPUdiagnostic: TCNepoch9 continuousalpha mean.00022797, GRUepoch12 .00043499; newcomponent CE down but finalmodel nearlyignores branch. V263 prepared directprior-relative bounded3 mass/no learnedmixturegate, same newencoders joint oneSWstage, fresh104/no262weights; held until262terminal/max2.

V262 bigru final SW74.122942685760, actualTOTAL80.612891868155; learned gain+0.000892016904, vs211-0.024679936867. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V262 tcn final SW74.122529455623, actualTOTAL80.612747237607; learned gain+0.000478786767, vs211-0.025093167004. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V262 FINAL TCN74.122529455623/best12/early16, GRU74.122942685760/best2/natural16; learnedinitial+.000478786767/+.000892016904, no meaningfulgain vsretained252. ActualTOTAL80.612747237607/80.612891868155/full144665finite/RAMbest/frozenexact; no independentdiskreload. Individual213.611151/177.972639s, peak938.692/1295.686MiB; pairwall222.291s includingstartup/data/initialeval/finalrestore, excludesprep. Architectureparamcounts differ98497/96513. Sampledtrainingcontinuous alpha nearzero onlydiagnostic, no validationfit. V263 launched directpriorrelative distribution/newencoders/no learnedmixgate/fresh104/oneSWphase/max2/80-14. Gains unproved, originalcurrentbest/default5/delivery untouched.

Unpromising best/last checkpoints deleted after assembly; manifest preservesSHA/paths/reasons and compactresults. No currentpositive/active source removed.

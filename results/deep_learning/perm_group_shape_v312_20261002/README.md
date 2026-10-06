# PERM grouped response shape v312

## Focus
Retain fresh291point stem, sparse selectors, two GLU blocks and original256-bin density readout. Jointly train the complete new body/readout plus two small grouped response banks. Only original154 prior loaded; no291/307/309 repairs. One added PERMphase, no width/depth/bin expansion. KAT-inspired rational activation migration, not full KAT reproduction.

## Predefined matched contrast
Each bank has eight groups of six channels, six numerator coefficients per group and four shared coefficients:52 parameters, two banks104. Body bank shared across both GLU value48 outputs BEFORE sigmoid-gate multiplication; separate bank transforms final48 feature before original readout. No LN/BN/extra affine or gates.
R=N_5(z)/(1+|B_4(z)|); P=N_5(z)+sum b_k z^(k+5), z=tanh(v/3). Actual map T(v)=v+.25*tanh(F_theta(z)-F_anchor(z)). Anchor buffers independently cloned fixed, a1=1 and b1=.1, other coefficients0, so both initial feature functions identical original291. Nonzero b1 avoids a permanently zero abs-denominator gradient. All coefficients used; high-degree P gradients may be poorly conditioned, function classes/optimization geometry differ. Same36570 stored trainable parameters versus historical29136466, capacity difference104 disclosed.
FP32 Horner arithmetic, denominator>=1 and bounded z/residual; no claim this guarantees optimization stability. Existing second-step actual group-gradient/finite/frozen/population/whole-best RAM restoration checks remain. Only all-new joint train, no coefficient-only tuning.

## Gains and closure
Prepared/untrained, gains unproved. Restore own wholePERM official-best, report DEFAULT/CONT/overall and R-minus-P. Retained291point89.5155547932572/CONT74.0010663397606. If rational CONT does not beat P, or absolute whole score does not beat291, close this setting; preserve local positive diagnostics honestly, no degree/group/bound/LR grid. No DEFAULTCE migration. Current optional8TOTAL80.69562059394032/default5 unchanged. Strict original fifth TOTAL>80.8+useful gain before same-config other folds; full OOF>82 incomplete.

## Sources and history
[Research with precise formulas and historical boundary](../cross_domain_research_20261001/PERM_GROUP_RATIONAL_RESEARCH_20261002.md).
[KAT primary paper](https://arxiv.org/abs/2409.10594) supports grouped learnable rational basis. Our anchored bounded GLU/readout use differs from its vision architecture/initialization; well-log benefit is unproved. Prior input PLE/RBF, target RQS/CDF, DCN/CORN and curveattention are distinct, no first-KAN novelty claim.

## Timing
Measure individual time/peak and same-work paired wall including data/initial/train/best restore, excluding preparation/CPUrecord. No automatic launch while current GPU slots occupied or claim custom-CUDA speed from pure PyTorch implementation.

V312 rational explicitly launched PID80048, original154 prior initialPERM89.405566626894/36570params; polynomial not launched yet. Actual joint encoder/readout/shape gradients monitored by training; no standalone preflight. No third GPU.

V312 final PERM89.515804404879, gain+0.110239547960, actualTOTAL80.651059259780; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds230.112487/peakMiB881.833496. No otherfolds.

V312 final PERM89.515098900854, gain+0.109534043935, actualTOTAL80.650812333371; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds144.507527/peakMiB847.802246. No otherfolds.

Final312 whole scores {'rational': 89.51580440487884, 'polynomial': 89.51509890085404}; rational-polynomial +0.000705504025; CONT paired +0.002335749759; strata {'rational': {'default': 97.49399611826219, 'continuous': 74.00288840401272}, 'polynomial': {'default': 97.49412903900604, 'continuous': 74.0005526542539}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 532.047s; no fullOOF or remaining folds.

V312 final: rational89.515804404879 vs polynomial89.515098900854; pairedwhole+.000705504025/CONT+.002335749759; versus291whole+.000249611622/CONT+.001822064252. Actual newoptional8TOTAL80.695707958008, delta+0.000087364068, strict808false. Retained as micropositive only, default5/deliveries unchanged; no extra phase/newgrid/otherfolds or robust gain claim. Current recipe here; old302recipe frozen. Rational230.112s/poly144.508s, totalwall532.047s incl sequentiallaunchgap/different GPUoverlap, not isolatedspeedranking.

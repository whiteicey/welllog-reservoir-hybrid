# POR numerical token PLE v305

## Exploration focus
A single migration of train-quantile piecewise-linear numeric embedding into the full curve-token encoder. Original151 protected prior; fresh302 rawbody/readout/attention in one jointly trained addedPORphase, no302checkpoint. Replace shared scalar Linear(1,32) with shared8basis->32; 9 curve-specific train-input quantile grids. Full communication, old206 bypass/loss/.25ramp5/seed/80-14 unchanged.

## Prior evidence / control
PERM276 already tested8segmentPLE versus scalar-repeat and gained+.0106716842 locally but remained below274 by.0000125369, unadopted. This is not firstPLE. POR302full78.030552794145 versus29078.023526555255, small+.00702623889. Completed302full is historical control; active params76816 versus76592 (+224, about0.29%), so mechanism+capacity change, not matched-capacity proof. Same initial151 predictive function because bridgezero and alpha0, not identical internal token initialization. New shared projection freshly seeded20261305, otherfresh302 initialization retained.

## Explored and verification
Prepared, gain unproved. Fit9quantiles only on original training raw points/curves, no target or validation statistics. Repeated edges lifted by max(trainingrange*1e-6,1e-5), adjustments recorded. Internal bases clamp0..1; first/last bases extrapolate linearly to preserve extreme amplitude instead of endpoint clipping. The first/last linear extrapolation follows the original numerical-embedding paper Eq.1; an all-clamped variant would be a different choice. Curves are original preprocessed9numeric inputs, no new DEVI/AZIM etc. Valid masked finalfeatures and originalsamewell context. Live ordinary source replay/frozen checks/finite losses/active group gradients/ownbestRAM replay, no redundant preflight. No independentdiskreload until qualification/finaldelivery.

## Gain
Unproved, only fifth. Compare restoredwholePOR-best against302full and290; actual144665 threehead assembly afterward. Retain current8stage80.695620593940/default5/oldOOFdelivery. StrictTOTAL>80.8 usefulgain before remainingfolds; fullOOF82 unmet.

## Not worth continuing
If no wholePORgain versus302full, close this token-PLE migration, no knotcount/endpoint/LR scan. If small positive, treat single-seed warn-onlyattention nondeterminism and224capacity confounding explicitly; not proof allpiecewise embeddings win. No frozenpredictionhead-only attempts or stacking302trainedphase.

## Timing
Record individual load/train/bestrestore and peak, paired/overlap samework wall separately; quantilefit counted. Max2GPU, newcode prep/CPUassembly excluded. No speed claim versus historical302 contention.

Explicit v305 POR PLE launched PID60056; original151 only/fresh302 architecture, 76816activeparams (+224). Train-only quantile token embedding and allnew rawencoder/readout jointly optimized in onePORphase. No302weights; 80-14/max2/max8 unchanged. Gain unproved.

Initialization metadata inherited from302 describing its grouped/full matched pair applies to that parent experiment only: only_scope_differs_in_matched_pair and registered_and_active_capacity_same do not claim matched capacity for305 versus302. Dedicated305 fields (+224/same_capacity_ablation=false) and this protocol govern this comparison. Active source/config remain frozen.

V305 final POR78.021651568327, gain+0.095270807603, actualTOTAL80.641056660275; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds232.126179/peakMiB1817.208984. No otherfolds.

Final305 POR78.021651568327, versus302-0.008901225818, versus290-0.001874986928; best/stop6/20. Adopted=False; +224capacity/nonconcurrent control/single-seed nondeterministicattention means no pure mechanism/robustness claim. Individual232.126s/peak1817.209MiB; trainquantilefit included. Whole-best144665RAM/frozenexact, not independentdiskreload, no fullOOF/remainingfolds. If negative close knot/endpoint/LR grid.

Research implementation distinction: [final source/history study](../cross_domain_research_20261001/POR_CURVE_TOKEN_PLE_RESEARCH_20261002.md). Canonical original Eq.1 supports our first/last extrapolation; repeatedquantileedge lift is our protection, whereas originalpaper removes zero-widthbins. No claim exact releasedimplementation reproduction.

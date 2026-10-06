# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW adaptive allocation v315

## Focus and hypothesis
Preserve the fresh297well input encoder, state readout, whole-well FiLM and density readout. Train them jointly with a bounded allocation module in one phase, loading only original104. Fixed mixtures retain at least 75% old prior mass after ramp, which may restrict local correction; this is a mathematical constraint, not evidence it causes the SW bottleneck.

## Predefined comparison and verification
Scalar bias versus conditional Linear48-to1 allocation; both zero initialized with identical original fresh initialization. alpha=.25*min(epoch/5,1)*(1+tanh(u/.3)), range0..0.5. Registered share capacity49 each; active1 versus49, so not a same-active-capacity ablation. Entire new body/state/FiLM/density jointly train; active65266 versus65314. All original losses, four direct continuous CE features, state bound4, train Q50, original grid, decoding and fixed split unchanged. No added loss, new inputs, old repair weights or stage chaining. Record actual alpha min/mean/max and real gradient groups in ordinary training. Whole SW selects checkpoint; compare DEFAULT/CONT at that checkpoint, not independently selected strata.

## Gains
Prepared, untrained, gain unproved. Primary whole control297well74.16003022301975, CONT17.502752767347182; stronger CONT25217.88486813991017. Optional8TOTAL80.69570795800789 remains unchanged. Qualificat

erature grids. Single seed, tiny positives not robust.
[Research](../cross_domain_research_20261001/SW_PRIOR_CORRECTION_LIMIT_RESEARCH_20261002.md).

## Timing
Maximum two actual GPU jobs; launch only after admission. Record each seconds/peak and pair wall including warmup, data and restoration. Preparation and CPU finalization excluded. Variable overlap is not an isolated speed comparison.

V315 snapshot16: scalarwhole74.131531/CONT17.703990 (+.201238CONT vs297 butwhole-.028499); conditionalwhole74.124596/CONT17.556323. Conditional earlyboundary best-restoring request; scalar keeps natural14patience. Noadoption; continuous/default tradeoff is retained evidence, not blanketnegative.

V315 conditional: SW74.131849364002, actualTOTAL80.616009205539; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V315scalar earlyboundary requested after32; selectedbest26 whole74.14209008848303/CONT17.657627078824376, absolute297whole-.017940/CONT+.154874, strong252CONT-.227241. Retain continuous/default tradeoff positivecontrol, not adoption or allocation grid. Terminal restoration pending.

V315 scalar: SW74.142090088483, actualTOTAL80.619593459108; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final315 whole scores {'scalar': 74.14209008848303, 'conditional': 74.13184936400198}; conditional-scalar -0.010240724481; CONT paired -0.059682158997; strata {'scalar': {'default': 98.73361820152378, 'continuous': 17.657627078824376}, 'conditional': {'default': 98.74490269587389, 'continuous': 17.597944919827714}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 557.720s; no fullOOF or remaining folds.

V315 closed: scalar74.14209008848303/best26/stop34 CONT17.657627078824376, selectedscalaralpha0.045009106; conditional74.13184936400198/best17/stop18 CONT17.597944919827714. Conditional-scalarwhole-.010240724481/CONT-.059682158997. Bothwholebelow297; scalarCONT+.154874 butbelow252. Retain CONTtradeoff evidence, noadoption/allocationgrid. Unequal adaptiveearlyclosure/active1vs49 explicitly notmatchedcapacity proof. Actual6assemblies80.619593459108/80.616009205539 below808. Pairwall557.720221s, individual552.561913/335.255871s/1318MiB; originalsourcefrozen and ownbestRAM144665exact, notindependentdiskreconstruction. V316 actual16564/reference37364 live, original104initialsame/80709, stagger227.14s. Retainedoptional8TOTAL80.695707958008 unchanged.



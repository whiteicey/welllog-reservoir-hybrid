# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# POR full-curve plus low-rank interaction v308

## Focus / positive parent evidence
Combine two scoped positive mechanisms by freshly training them together in ONE PORphase, original151 protected prior only. 302full POR78.030552794145, full-grouped+.006687052660; 306 anchoredcross-additive+.004795594278/continuous+.013790145627, although306absolute78.025028 below302. Preserve complete302 freshfullcurve encoder/3MLPbranches/readout, add two zero-init rank8 updates after stem's curvecorrection; no302/306 trainedweights. OldDCN/curveattention history already documented; not a new algorithm or stackedstages.

## Explored / fixed comparison
Both candidates have full9curveattention; at48stem output (parenthook first), h initialized to correctedstem, anchor=sharedLayerNorm(h0), delta_l=U_l V_l LayerNorm(h_l)+b_l, h_l+=.1*(anchor*delta_l cross OR delta_l additive). SharedLN actively participates in lowrankinput in BOTH, avoiding unused96params inadditive. TwoUV rank8=1632 plusLN96=1728, total78320 versus30276592. U/bzero/Vrandom seeded20261308; bothinitialhiddenfunctions exactly match fresh302full; sharedtokens/readout original302 initialization unchanged. Effectiveconditionalfunction differs later, intendedvariable. Unlike306, preserve3branchMLP and curve module, and normalize lowrankinputs/anchor; transfer not exact306reproduction. Fouroriginal302auxfeatures andreadoutlossunchanged,

mbination, no rank/layer/scalegrid. Fairlocalgain can remain scoped evidence but nevercallabsoluteadoption. No further chains of savedrepairweights or extraheadstages. Lossdecrease notofficialgain.

## Timing
Individual data/initial/train/bestrestore/peak plussameworkpairwall/actualoverlap; codeprep/CPUrecord excluded. Max2GPU explicitlaunchonly afteractiveslots free; not an automaticqueue. Historicaldifferentcontention prevents speedranking.

Prepared v308, NOT launched: preserve allfresh302full modules; add1728params two rank8 sharedLNupdates, cross/additive same78320params/init. Allnewbody/attention/cross/readout inONEjointPORphase fromoriginal151only, no302/306trainedweights. Current307 occupies2GPU slots. Pending explicitlaunch afterslots, score/strata/pair/time/actualcomposition; no autoqueue, no gainyet.

Explicit308 joint pair launched: cross PID44656, additive PID76968. INITIAL77.926380760724 and78320activeparams both confirmed from live logs; original151 only, parent302architecture retained/no302 or306 weights, allnewbody/attention/cross/readoutonephase. Max2GPU, current8stage80.695620593940 unchanged. Pending finalpair/strata/time/actualassembly.

V308 final POR78.021345596608, gain+0.094964835884, actualTOTAL80.640964868759; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds643.216918/peakMiB1814.011230. No otherfolds.

308cross final: POR78.021345596608, best11/stop25, versus retained302 -.009207197537; extra1728params group actuallylearns (firstgroupnorm .00158365), all144665RAM/frozenexact. Individual643.217s/peak1814.011MiB underdualGPU. 308additive stilllive andrecentbest78.028308460582 at29; pairfinalpending, do not call entirecombination negative yet. 309actualdefaultbin0training launched aftercrossPIDmissing, no thirdGPU.

V308 final POR78.028308460582, gain+0.101927699859, actualTOTAL80.643053727952; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds976.572693/peakMiB1813.394043. No otherfolds.

Final308 cross78.021345596608/additive78.028308460582; paired-0.006962863975; absoluteadoption=False, selectedadditive. Same78320params/init/fullcurve, onlyanchorproductdiffers. No oldrepairweights, onePORaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall983.066s, individuals{'cross': 643.2169180000201, 'additive': 976.5726931000245}. Ifbothbelow302 close combinationgrid; scopedgainnotabsoluteadoption. Bestcheckpoint blocknorms recorded to distinguish zero-init function from trained interaction.



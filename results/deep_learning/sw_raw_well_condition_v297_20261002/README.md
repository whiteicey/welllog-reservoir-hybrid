# SW input-only bank conditioning of new raw encoder v297

## Focus
275/285/289/294/296 newraw encoders ignore well_inputs, although frozen104 prior uses it. A small learned whole-well condition can provide lithology/fluid/background input context to the new48rawfeatures. This is NOT first learnedwellcontext: v46/104 inherited learnedWellEncoder, and277/278 copied full104 encoder trained its condition but achieved74.123330638258 below252, with budget/OOM caveats and no bank-vs-reference ablation. This trial targets the new directcategorical raw branch only.

Original104 frozenprior remains on actual same-well bank in BOTH arms. Newcondition compares actualbank vs positionwise64traininginput reference; oldprior never changes. Nine originalcurves from exact source names only, bank128stratified inputpoints perwell; no labels/IDs. Newbank9->32->32, mean/std/max->32->96 outputs boundedgamma/beta .25, zerooutputinit; modulate newstem48 before3existingpoint branches. Local206/missingflags remain unchanged; conditionbranch9doesnotsee missingflags. Allnew rawencoder/density/state/condition jointlytrain3e-4; original104 only/no289/277/278trainedweights, oneSWaddedphase/max8.

## Explored and verification
Prepared/untrained, GPUheld296/295 studies. Same65265params/seed20261297/initialbody and zeroFiLM for reference/well; only conditioninputscope differs. BalancedBCE explicitlyforced for both armnames, share.25ramp5/statebound4/directcontinuousCE1/conditionalRPS/80-14/min10/b64/64steps/scoring/Q/split unchanged. Initialalpha0/gatepatch0 reproducesoriginal104; ordinaryinitialeval guard noextraGPUpreflight. Reference buffer stored in checkpoint, fitted64traininginputbanks only; heldout uses ownbank at inference, no cross-heldout input fit. See inputbank helper fullcv_context_v97.add_well_bank and research background when finalized.

## Gain
Unproved. Primary finaldelta well-reference; absolute compare SW28974.1537264289993 and continuous25217.88486813991017. Must restore overallSW-best before actual144665assembly/strata; continuous-onlydiagnosis is not newcheckpoint selection. FullOOF82 unmet/strictfifth80.8meaningfulgain before otherfolds.

## Not worth continuing
No result yet. If actualwellbank failsreference or bothbelowretained withoutcontinuousbenefit, close this smallFiLM setting, no descriptor/boundgrid. Globalformation mixtures can obscurelocalfluidconditions; Rw/lithology is not identifiable fromthese statistics alone. Missing/imputation distributions may become shortcuts; no assertion curvebank is causal or anomalylabel.

## Timing
Separateindividualprep/initialeval/train/finalrestore fromsameworkpairedwall, measureactualoverlap/peak. Codeprep/CPUcompositionexcluded, max2GPU/noOOM. Frozenprior already computesbankfeatures; newencoder adds measured workload, not claimedfree.

V297 well final SW74.160030223020, actualTOTAL80.625872506196; learned gain+0.038277533251, vs211+0.012407600393. Six optionalphases, default5; new65265raw_wellencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

## Verified well-arm completion; reference control active
Well completed at best17/stop31, SW74.16003022301975; own-best RAM replay/frozen weights and actual144665 CPU composition checked, not independent disk reload. Optional8 POR290/PERM291/SW297 well TOTAL80.69351272227345, +0.00220632790714 versus prior optional8. No >80.8 qualification, no remainingfolds/fullOOF; default5 and earlier deliveries unchanged. Saved separate WELL_EIGHT_STAGE_RECIPE.json and predictions; retain well positive checkpoint while reference arm completes.
Default SW98.82679621708131 versus28998.72311025198368, but continuous SW17.502752767347182 versus28917.72012680313723 and25217.88486813991017. Thus this is a default-state gain, NOT improved continuous readout. Whole-well causality remains unproved until same-capacity reference comparison. Individual563.871727s/peak1280.802MiB includes data prep/initial/train/finalrestore; pair wall pending. Reference PID75804 launched after well exit, overlaps296pooled; at most2GPU. No descriptor/bound grid based on this small absolute result.

V297 reference final SW74.154114272860, actualTOTAL80.623801923640; learned gain+0.032361583091, vs211+0.006491650233. Six optionalphases, default5; new65265raw_wellencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final matched contrast
Well SW74.160030223020, reference74.154114272860; matched well-minus-reference+0.005915950160. Continuous well17.502752767347, reference17.463982441198, matched+0.038770326149; BOTH must be compared to retained28917.72012680313723/25217.88486813991017 before declaring continuous gain. Optional8 actualTOTAL80.693512722273, strict808 false; separate positive candidate and matched control retained, no additional stages/folds. Individual well563.872s/reference366.630s, pairwall969.381s incl stagger and overlap against296, not speed ranking. Bank scope evidence applies only to newraw FiLM, not globally to all lithology/fluid modeling. Further descriptor/bound sweeps not justified by this small gain; next priority substantive continuous representation.

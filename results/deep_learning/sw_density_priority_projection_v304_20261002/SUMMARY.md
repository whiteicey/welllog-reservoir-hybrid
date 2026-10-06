# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW density-priority shared gradient projection v304

## Focus / prior evidence
301 detached all state gradients into newrawbody and lost SW-.004822/continuous-.225620 versus289. Instead preserve aligned state supervision and project only its conflicting component: gs'=gs-min(dot(gs,gd)/||gd||²,0)*gd; sharedgradient=gd+gs'. Densitytask is mixedCE+conditionalRPS+directcomponentCE, statetask exactlytrainQcalibratedbalancedBCE/3. Private gate/density heads retain original total-loss gradients, original forward/mainlossvalues/optimizer/schedule/inference unchanged. Allnew encoder/readout/state jointlytrain in ONE addedSWphase from ORIGINAL104, no289/301weights, same57617capacity/init/ .25share/statebound4/80-14.

Not first PCGrad: posterior_pcgrad_loss projected three expert targets randomly; PERMv68 BCE/NLL symmetricprojection had about43.4% conflict but fullscore88.827250 unchanged/failedadoption. CurrentSW275->289 uses summedloss; this is a single asymmetric state->density gradient-priority migration, not standard random/symmetric PCGrad reproduction, not more expressive network. Historical289 completeconfiguration reused as control, not concurrentlyrerun. A graph-gradient decomposition includes all losses; projection restricted to group0 newrawstem/branches/fuse, no frozenprior parameters.

## Explored and validation
Prepared/untrained. Ordinaryinitial104 replay; original loss

weight grid. State features can help density even with locally opposed gradients; perbatch noise versus generalization conflict differ. Negative oldPCGrad/currenttrial not family-wide proof. Preserve active/key/control weights; redundantnegative cleanup presentlypolicyblocked.

## Timing
Measure individualdata/initial/train/finalrestore and peak, including extraautogradwork; record epoch time against historical289 with contention difference, no speedranking or claimedfree projection. Max2GPU; codeprep/CPUassemblyexcluded.

Explicit v304 launched PID81008: new raw SW backbone and readout jointly train from original104 only, one added SW stage. Keep aligned state gradients; project only conflicts with density. No extra parameters, no changed inference, max2GPU/80-14. Gain unproved; pending whole-best/strata/conflict/cost results. Research: ../cross_domain_research_20261001/SW_ASYMMETRIC_PCGRAD_RESEARCH_20261002.md.

Metadata clarification: frozen PLAN inherited primary_control_SW252 / fusion_control / old hypothesis from289 ancestry; actual v304 controlled change is directed shared-body gradient projection against completed289 SW74.1537264289993. Network/loss/parameter source and new dedicated protocol above are authoritative; no fusion-type or state-bound sweep is running. Prepared_not_launched in frozen PLAN is preparation-time snapshot; LAUNCH_STATUS and runtime_status establish actual launch.

V304 density_priority final SW74.153254610940, actualTOTAL80.623501041968; learned gain+0.031501921172, vs211+0.005631988313. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final gradient-routing outcome
SW74.153254610940/best32/stop46, versus289-0.000471818059; continuous17.713332876240, versus289-0.006793926897/252-0.171535263670. Not adopted; no priority/threshold/BCE grid. Actualencoder/readout/state nonzerograd/frozen/RAMbest144665assembly; no independentdiskreload. Individual401.474s/peak1373.719MiB, single GPU, historical contention differs/no speedranking. Decoded atom actionfraction is a posterior decision diagnostic, not gateprobability/classifier accuracy; no roworacle or stratum-specific model selection. Currentrecipes unchanged; finishedcleanup remains blocked by automaticpolicy, no deletionperformed.

Timing correction: mostly a singleGPU training task, but POR305 launch overlaps the final46th epoch/restoration briefly. Not completely isolated. Individual401.474s/launch-to-report405.787s, measured peak1373.719MiB; no historicalspeedranking.



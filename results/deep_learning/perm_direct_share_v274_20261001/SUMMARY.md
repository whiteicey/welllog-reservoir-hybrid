# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v274 stronger direct component during joint representation training

## Focus
271repeat effective recipe learned new raw featureencoders and categoricalreadout jointly and beat268PERM by.02355. Retain exact57568 architecture/init/seed/directcomponentCE/halfrow-halfequalwell mixtureNLL/jointRPS1; change fixed finalalpha.1->.25 withsame5epochramp. Original154prior still75percentconditionalweight. This is another full jointencoder/readout trainingphase, not inference interpolation search or head-only patch. No271trainedweights loaded.

## Explored and validation
Prepared held272/273 twojobs. Historical271repeat is fixedsame-protocol control; do not rerun identical deterministiccontrol withoutreason. No claim wholetrainingtrajectory unchanged as higheralpha changes mixtureloss fromfirstepoch. Original256window/206+relative/Q50/fold5/seed/80-14/b64/64steps/nativebins/actions. Schedulebuffer saved inbest/last and explicitresume; initialdiscretization tinychange separatelyrecorded. Finalactual144665compose/RAMbestproof required; no independentdiskreload claim.

## Gains
Unproved. Predefine candidatePERM minus271repeat89.43939135949043, plus actualTOTAL withPOR151/SW252. OnePERMaddedstage replaces optional271phase,6alone/7withSW252, no chainedrepairweights. StrictTOTAL80.8+meaningfulgain beforeotherfolds; fullOOF82unmet/default5/deliveryunchanged.

## Not worth continuing
Multiscale 

w-halfequalwell mixtureNLL/jointRPS1; change fixed finalalpha.1->.25 withsame5epochramp. Original154prior still75percentconditionalweight. This is another full jointencoder/readout trainingphase, not inference interpolation search or head-only patch. No271trainedweights loaded.

## Explored and validation
Prepared held272/273 twojobs. Historical271repeat is fixedsame-protocol control; do not rerun identical deterministiccontrol withoutreason. No claim wholetrainingtrajectory unchanged as higheralpha changes mixtureloss fromfirstepoch. Original256window/206+relative/Q50/fold5/seed/80-14/b64/64steps/nativebins/actions. Schedulebuffer saved inbest/last and explicitresume; initialdiscretization tinychange separatelyrecorded. Finalactual144665compose/RAMbestproof required; no independentdiskreload claim.

## Gains
Unproved. Predefine candidatePERM minus271repeat89.43939135949043, plus actualTOTAL withPOR151/SW252. OnePERMaddedstage replaces optional271phase,6alone/7withSW252, no chainedrepairweights. StrictTOTAL80.8+meaningfulgain beforeotherfolds; fullOOF82unmet/default5/deliveryunchanged.

## Not worth continuing
Multiscale pool271provided noextra gain; no scale/width expansionhere. Existing newcomponent only10percent may capbenefit but notproven bottleneck; stronger share can increase bias and harm officialactions. DirectCEdownward notutilityproof. If .25 loses271control, stopthisshareincrease, no automaticalpha ladder or post-hocvalidationalpha search.

## Timing
Individual/actualwall/overlap/peak, startup/data/initialeval/finalrestore included/prep/CPUassemblyexcluded. Maximum2GPU; launch only after verified slotfree, no automaticqueue.

V274 final PERM89.482758477027, gain+0.077193620108, actualTOTAL80.639493185032; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds212.270299/peakMiB788.226562. No otherfolds.

V274 FINAL alpha25PERM89.482758477027/best8/natural22, vsalpha.1 samebody/loss/init/control+0.043367117537; directrawencoder+readout jointlylearned, no inferencealpha scan/no oldrepairweights. Actual144665/RAM/frozenproof/no independentdiskreload; actualsevenphaseTOTAL80.649109567911/previous7delta+0.015178491138, strict808false/nootherfolds/fullOOF82unmet/default5/deliveryunchanged. Retainpositiveweights/control271. Individual212.270299s/788.227MiB/launchwall222.251s inclstartup/data/initialeval/finalrestore/excludesprep, overlapdifferenttasks/no equalworkserialclaim. V275 SWnewrawencoder/directhead mix/tilt bothGPUactive, allnewbackbones+readoutsjointtrain/max2/80-14/oneSWphase.



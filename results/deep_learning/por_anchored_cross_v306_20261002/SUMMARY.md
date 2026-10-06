# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# POR anchored low-rank raw encoder v306

## Focus / source / prior history
One bounded migration of DCN-style input-anchored low-rank interactions into a fresh POR raw encoder, original151 prior frozen. [Source/history research](../cross_domain_research_20261001/ANCHORED_LOWRANK_RAW_RESEARCH.md) identifies original DCN-V2 and Google official implementation. Old batchensemble_cross already used anchored interactions and scored fifth79.448021/POR75.442844; v152 product lost.008878; v192 multiply locally+.007156 but low absolute. Not firstDCN and no family-wide novelty claim.

## Explored / predefined pair
Fresh207->48 LayerNorm; two rank8 blocks hl'=hl+.1*(h0*UVhl+b anchored OR UVhl+b additive). Exact implementation cross=h+.1*h0*(UVh+b), additive=h+.1*(UVh+b). UV has no intermediate activation; U/b zero, V random. Pair params/init/function equal, effective polynomial interaction family intentionally differs. concat[h0,h2]->48 GELU then residual48->96->48MLP, shared48->256readout. OriginalthreeMLPbranches entirely replaced; neither302attention nor305PLE included, no290/302/305trainedweights. Body25728/readout12544=38272params. Four auxiliary directCE feature states h0/h1/h2/fused retained same averaging/coefficient1; newrepresentation/placement changes versus290/302, so historicalabsolute comparison is not isolatedcrossattribution. Only cross-additive pair supports that attribut

xact via zeroU/b. V firststepgradient0 expected untilUupdates; existing wholebody/readout gradient checks and laterlivelearning, not a requires_grad-only learning claim. Finite loss/livefrozenchecks/atomic fullstates/bestRAM144665assembly; no repetitivepreflight/review or independentdiskreload before qualification.

## Gain
Prepared, no result. Compare cross vsadditive ownbestwholePOR, thenabsolute302full78.030552794145/default5/retained8TOTAL80.695620593940. Actualcomposition only afterfinal, no stratumcheckpointselection. Original151 only, oneaddedPORphase/total<=8/strict808 usefulgain before otherfolds, fullOOF82 unmet. Olddeliveries unchanged.

## Unworthy continuation
If cross<=additive, close rank8/twoblock/.1 setting, no rank/layer/scale grid. If fairpositivebutabsolutechampionnegative, retain scoped evidence without adoptingwholecandidate. Multiplicationcanamplifyhiddenoutliers, LN/.1 not a guarantee; earlystop evidence if hopeless. No labelretrieval/teachers/pretrainedexternalweights or historicalroutingmix.

## Timing
Individual load/initial/train/restored-best and peak; samework2arm overallwall andactualoverlap separately. Max2GPU explicitlaunch, notautomaticqueue. Codeprep/CPUrecord excluded; compare no historicalspeedranking withdifferentcontention.

Explicitv306 pair launched: cross PID61712, additive PID31260. Both freshbody38272params/newreadout joint, original151 only, max2GPU. Initialsourcefunction77.926380760724 and actualactivecount confirmed, no extra stages or oldweights. Pending terminalpair/strata/realwall.

V306 final POR78.025027984200, gain+0.098647223476, actualTOTAL80.642069585037; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds421.501598/peakMiB982.318359. No otherfolds.

V306 final POR78.020232389922, gain+0.093851629198, actualTOTAL80.640630906753; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds421.089864/peakMiB982.427734. No otherfolds.

Final306 cross78.025027984200/additive78.020232389922; paired+0.004795594278; absoluteadoption=False, selectedcross. Same38272params/initialfunction, effective interaction family differs. No oldrepairweights, onePORaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall443.469s, individuals{'cross': 421.50159800000256, 'additive': 421.0898638999788}. If cross<=additive close rank/layer/scalegrid; localpositivebutabsolutebelowchampion not adopted. Bestcheckpoint blocknorms recorded to distinguish zero-init function from trained interaction.



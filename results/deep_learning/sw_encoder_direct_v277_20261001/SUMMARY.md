# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW v277 directly supervised copied encoder scope

## Focus
Late vs full copied104 encoder joint training, direct256categorical readout48 width. Frozen original104 prior/gate, fixedalpha0?.1 in5epochs. Late updates detail/fuse/norm; full updates all104 feature backbone/conditioning, old output frozen unused. Same copiedweights/readout seed20261277/initialfeatures/loss/decoding; different trainablecapacity is deliberate scope comparison. Both satisfy backbone+readout training. No trained252/275 weights.

## Explored and validation
Prepared for direct originalfold5 training. Original275 auxiliary CE already continuous-only normalized, NOT default-dominated auxiliary loss. Copy effective275 mixed hard/tolerance5pctCE+conditionalRPS1 plus directcomponentCE1 on one hidden feature, not four rawbranches. Existing260 trained originalGaussian+small zero-init mass correction; this is fresh nonzero categorical component and direct supervision independent of alpha. DropKey/dropout eval with gradients to avoid frozen-prior stochastic changes. Full144665 best restoration and actualassembly required; RAMreplay not diskreload.

## Gains
Unproved. Predefined full-minus-late, absolute SW25274.14922820781207 and275mix74.13723635928496. Each oneSWaddedphase, six alone/seven with retainedPERM274, total<=8. Strict fifthTOTAL>80.8+meaningfulgain before otherfolds; fullOOF82 unmet. Original default5/

epared for direct originalfold5 training. Original275 auxiliary CE already continuous-only normalized, NOT default-dominated auxiliary loss. Copy effective275 mixed hard/tolerance5pctCE+conditionalRPS1 plus directcomponentCE1 on one hidden feature, not four rawbranches. Existing260 trained originalGaussian+small zero-init mass correction; this is fresh nonzero categorical component and direct supervision independent of alpha. DropKey/dropout eval with gradients to avoid frozen-prior stochastic changes. Full144665 best restoration and actualassembly required; RAMreplay not diskreload.

## Gains
Unproved. Predefined full-minus-late, absolute SW25274.14922820781207 and275mix74.13723635928496. Each oneSWaddedphase, six alone/seven with retainedPERM274, total<=8. Strict fifthTOTAL>80.8+meaningfulgain before otherfolds; fullOOF82 unmet. Original default5/delivery preserved.

## Not worth continuing
If full does not beat late nor retained252 after best restore, close this lr1e-5/directcategorical/alpha.1 config; no blindLR or width grid. A negative does not prove all backbones useless. Continuous/default strata are diagnostics only, never row-wise selection. No full replacement/extra stages/teacher/labels-as-input.

## Timing
At most two concurrent GPU tasks. Record actualpair wall and individual elapsed/peak; include startup/data/initial/final, exclude codeprep/CPUassembly. No unchangedcontrol rerun beyond this genuinely changed matchedreadout pair.

V277 launched and actual processes verified: latePID28564/fullPID47148; identical initialSW74.121752689769, trainable144816/861424 respectively. Joint backbone/readout, no teacher or prior repair weights. CUDA backward warnings mean no bitwise training reproducibility claim. Terminal results and assembly pending.

V277full failed initial training CUDA OOM (no trained result). Late remains active; newV278 full checkpointed source from original104, same function/init/loss/budget prepared.

V277 late final SW74.123123629860, actualTOTAL80.612955198590; learned gain+0.001370940091, vs211-0.024498992767. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

Closed paired scope: lateSW74.123123629860 natural16, fullSW74.123330638258 bounded13; full-late+0.000207008398, below252-0.025897569554, no adoption. Actualassembled144665/RAMreplay/frozenproof, no independentdiskreload. Budgets differ, not fullconvergence causalproof; close this fixedLR/alpha/readout scope only. Unpromising checkpoints may be cleaned after assembly.



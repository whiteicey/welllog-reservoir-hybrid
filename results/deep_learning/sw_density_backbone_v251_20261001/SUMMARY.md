# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v251 continuous-density backbone adaptation

## Focus
Original104 duplicated into immutable default-state network and continuous-value network. Head arm freezes continuous body; body arm trains it at3e-5 plus same12609param mass/background readout at1e-4. Same initialization/head/dataset/own halfhard-halftolerance binCE; AdamW with SAM0 in BOTH arms. Eval mode disables DropKey while gradients remain enabled. This changes optimization from246SAM.1, so compare new head control first, not pure attribution versus246.

## Explored and validation
Prepared, held until250 terminal. Gate logits from immutable104 atom network, continuous prior differentiable finiteCDF. Original256bins/800actions/64trainingwells/144665validation/Q50 unchanged. Body arm updates existing845913parameters including unused gate rows with weight decay; predictions only use immutable atom gate, unused rows have no supervised gate gradient. Original source artifact unchanged. Final RAM-best restore/frozen atom/actual composition required; no independent disk-reload claim.

## Gain
Unproved. Body-minus-head matched initialization/loss, different trainable scope and two LR groups. Absolute retainedSW24674.149145254062. Body and newhead train together ONE extra SW phase, six total/seven with retained PERM addition. No trained246/250 weights loaded.

## Not worth continuing
242 only final globalFF coadapt failed;25

m gate, unused rows have no supervised gate gradient. Original source artifact unchanged. Final RAM-best restore/frozen atom/actual composition required; no independent disk-reload claim.

## Gain
Unproved. Body-minus-head matched initialization/loss, different trainable scope and two LR groups. Absolute retainedSW24674.149145254062. Body and newhead train together ONE extra SW phase, six total/seven with retained PERM addition. No trained246/250 weights loaded.

## Not worth continuing
242 only final globalFF coadapt failed;250 bounded terminal residual currently no coupling benefit. Full continuous-side adaptation is a broader distinct scope. No further full-body LR grid if no gain. Predefined early stop allowed after10 if recovery implausible.

## Timing
Record individual and concurrent process-creation-to-final-report wallclock, peakMiB. Includes startup/data/model/initialeval; excludes code prep and CPU assembly. No serial speedup claim.

251 actual launch headPID25000/bodyPID42372; both initial74.121701580675. Head12609params epoch1 best74.133401970608, body858522params epoch1score73.946483489217/no gain yet. Live/max2; peaks667.695/1464.630MiB, noOOM. Body backward emits interpolation/attention nondeterminism warn_only; no bitwise rerun reproducibility claim. Original atom remains frozen and full source checkpoint unchanged; early snapshot not final evidence.

V251 head final SW74.133401970608, actualTOTAL80.616552617852; learned gain+0.011700389934, vs211-0.014220652019. Six optionalphases, default5; one-stage continuous-backbone versus frozen-body readout comparison.

V251 body final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; one-stage continuous-backbone versus frozen-body readout comparison.

251 final head74.133401970608 best1/stop15 naturalpatience14; body74.121701580675 best0/stop12 earlytermination/no learnedgain. Body-minus-head -0.011700389934, unequalbudgets and backbone CUDAwarn_only nondeterminism disclosed; no convergence/globalbackbone rejection. RAMrestore/frozen atom/actual144665composition done. Positive head kept; negative body checkpoints removed withSHA manifest. Concurrent pairwall 227.838s; individual193.433/220.512s, includesstartup/model/data/initialeval, excludesprep/CPUassembly, no measuredserialspeedup. Close current3e-5continuousbody+mass/AdamW/SAM0 setting, no LRgrid.252joint/conditionalCDFsupervision new sibling original104 pair launched onlyafter251PIDs ended; max2/oneSWaddition/nootherfolds. Bestoptional7 unchanged80.624828025525.



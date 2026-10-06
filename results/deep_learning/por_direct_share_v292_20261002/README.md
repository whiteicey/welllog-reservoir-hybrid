# POR joint raw encoding share50 v292

## Focus
Continue proven joint rawencoder/directdistribution route: v287 share10 POR77.952002347840; v290 share25 POR78.023526555255, same original151 parent. Predefine share50 to test whether new component has more usable signal. Original151 prior/gate frozen; new57568rawencoder/readout jointly3e-4, same directCE1/physicalmixedCE/jointRPS1. Only maximum mixture schedule changes .25 to .5, ramp5 unchanged. No287/290 trained weights.

## Explored and verification
Prepared, untrained. Same data/splits/Q/scoring/seed/80-14/min10/64steps, own labels, selected mix_alpha must restore with best state. One added POR phase, final8 maximum; old recipes preserved. Same budget control290; higher amplitude also changes ramp trajectory, so best early alpha is not proof of fixed .5 benefit.

## Gain
Unproved. Primary compare final POR minus29078.023526555255; absolute compare prior15177.926380760724. Restore best and actual combine before gain claims. Strict fifthTOTAL>80.8 plus meaningfulgain before otherfolds; fullOOF82 unmet.

## Not worth continuing
If stronger share fails retained25, close current share escalation rather than blindly grow .75/1. Best checkpoint and compact evidence preserved according to result. No negative mechanism stacking.

## Timing
Separate individual training time and whole batch wall, actual concurrency/memory, prep/initial/final evaluation included; code preparation/CPU composition excluded.

V292 final POR78.008790775802, gain+0.082410015078, actualTOTAL80.637198422517; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds195.790577/peakMiB981.986328. No otherfolds.

V292 finalPOR78.008790775802/best3/stop17, versus290-.014735779454, selectedalpha0.30000001192092896 not .5. Rejectshare50 escalation, retain25; no.75/1grid. Actualassembly retained, redundantbest/lastdeletedSHAmanifest, positives151/287/290 untouched. Nootherfolds/fullOOF82unmet.

# perm_conditional_rps_v244

## Focus
Matched241 normalized continuousNLL; replace joint allrowRPS by conditional continuous-only normalizedCDF RPS1. Original154 gate/positive head frozen; no default-label gradients into conditional distribution.

## Explored and validation
Prepared/untrained. Matched control architecture/seed/optimizer/budget/zero initial repair; original positive parent frozen. Same original scoring/actions; actual144665 combination after restored-RAM replay/frozen check.

## Gain
Unproved. One headaddition/six alone; no loading trainedcontrol repair. Max2 GPU, measure individual and batch wallclock separately.

## Not worth continuing
242 coadapt currently below initial at7, no final claim yet. This new matched supervision direction avoids more width grids; if no learning benefit close particular loss, not whole conditional modeling. Preserve bestdefault5/optional7.

V244 final PERM89.413440514835, gain+0.007875657916, actualTOTAL80.615231898265; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds264.101818/peakMiB791.250000. No otherfolds.

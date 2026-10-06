# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# sw_decision_rank_v243

## Focus
Retain219 halfhard/halftolerance bin loss. Add utility-regret-scaled margin ranking of train-label-optimal action against current predicted winning action: hinge(.1*true_regret-score_gap), weight1. Gate frozen, bound.5.

## Explored and validation
Prepared/untrained. Matched control architecture/seed/optimizer/budget/zero initial repair; original positive parent frozen. Same original scoring/actions; actual144665 combination after restored-RAM replay/frozen check.

## Gain
Unproved. One headaddition/six alone; no loading trainedcontrol repair. Max2 GPU, measure individual and batch wallclock separately.

## Not worth continuing
242 coadapt currently below initial at7, no final claim yet. This new matched supervision direction avoids more width grids; if no learning benefit close particular loss, not whole conditional modeling. Preserve bestdefault5/optional7.

V243 final SW74.129922413897, gain+0.008169585739, actualTOTAL80.615334773003; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds293.088040/peakMiB908.494629. No otherfolds.





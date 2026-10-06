# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM local sparse curve reacquisition v293

## Focus
Following291point89.515554793257 vswindow89.510323066916 (same36466params/fair+.005231726341), test intermediate conditioning range. Same two sparsemax/GLU body and categoricalreadout, only pointselectorlogits become masked17point localmean. Dense representation and selected curves remain per-point. Original154 prior only, no291trainedweights, one addedPERMphase. All new encoder/readout train jointly.

## Explored and verification
Prepared/untrained. Same seed20261291/init/36466params/zero selector outputs/uniform initialmasks, sameauxdirectCE1/mixedNLL/RPS/alpha.3ramp5/Q/split/scoring/80-14. Localmean excludesinvalidpadding/no cross-well. Valid masks use existing inputonly boundaries. This is a new same-parent sibling, not second repair stage.

## Gain
Unproved. Primary compare finalscore against291point89.515554793257; absolute old28189.498908539006. Restore best/actual144665 composition required before gain claim. FullOOF82 unmet, fifthstrict80.8+meaningfulgain before otherfolds.

## Not worth continuing
If local17 does not improve point, close this local-selector range; no automatic window-size sweep. Priorpoint positive preserved. No claim sparse selection generally fails.

## Timing
Individual trainer prep/initialeval/training/finalrestore included; CPUcomposition/codeprep excluded. Record actualoverlapwall/peakmemory

# PERM local sparse curve reacquisition v293

## Focus
Following291point89.515554793257 vswindow89.510323066916 (same36466params/fair+.005231726341), test intermediate conditioning range. Same two sparsemax/GLU body and categoricalreadout, only pointselectorlogits become masked17point localmean. Dense representation and selected curves remain per-point. Original154 prior only, no291trainedweights, one addedPERMphase. All new encoder/readout train jointly.

## Explored and verification
Prepared/untrained. Same seed20261291/init/36466params/zero selector outputs/uniform initialmasks, sameauxdirectCE1/mixedNLL/RPS/alpha.3ramp5/Q/split/scoring/80-14. Localmean excludesinvalidpadding/no cross-well. Valid masks use existing inputonly boundaries. This is a new same-parent sibling, not second repair stage.

## Gain
Unproved. Primary compare finalscore against291point89.515554793257; absolute old28189.498908539006. Restore best/actual144665 composition required before gain claim. FullOOF82 unmet, fifthstrict80.8+meaningfulgain before otherfolds.

## Not worth continuing
If local17 does not improve point, close this local-selector range; no automatic window-size sweep. Priorpoint positive preserved. No claim sparse selection generally fails.

## Timing
Individual trainer prep/initialeval/training/finalrestore included; CPUcomposition/codeprep excluded. Record actualoverlapwall/peakmemory/max2GPU.

V293 final PERM89.514219912256, gain+0.108655055337, actualTOTAL80.650504687362; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds180.873825/peakMiB788.697754. No otherfolds.

V293local17 finalPERM89.514219912256/best5/stop19 vs291point-.001334881001. Same36466params/init/loss and prior154; only selectorlogits pooling changed. Rejectlocal17/closewindowgrid, point positivekept. Redundantbest/last cleanedSHAmanifest afteractualassembly; nootherfolds/fullOOF82unmet.



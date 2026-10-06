# POR curve-token width128 v337

## Exploration focus
Test whether the final positive width64 curve interaction can gain from width128. Change only fresh nine-curve attention: numeric1?128, identity9?128, two four-head attention blocks with FF256, zero128?48 bridge. Raw encoder and readout stay48; all fresh encoder/attention/readout jointly train. Only original151 checkpoint reused; no335/302 trained repair weights. 272560 attention /330128 active parameters. One added POR phase; six phases with default PERM/SW or at most eight with retained336/326.

## Controls and verification
Historical335: POR78.04221263500206 / CONT41.724510830138236, best28 / natural42. Prespecify best-through42 during ordinary80/pat14 training. A later best is an absolute candidate result, not equal-work gain; earlier stopping must disclose smaller work. Original fold/seed/Q50/score and loss/LR unchanged. Same seed with different weight shapes does not yield matched hidden initialization. RAM best replay and frozen signature checked at terminal; independent disk reload only qualification/delivery. No per-trial review/preflight.

## Gains
Prepared, untrained; no score or gain established. Retained8 TOTAL80.70344014130356 remains unchanged. Adoption requires positive whole-POR and CONT versus335, followed by actual combined validation. Strict80.8 and fullOOF82 unmet.

## Limits and closure
Bridge rank already at most48 for64;128 still48, so this is not further bridge-rank relief. Estimated attention forward39.454G MAC/b64L256 ~3.92x64, not measured time/peak. BF16 token36MiB/QKV108MiB/FF72MiB per block are not total GPU peak. Measure actual time, overlap and peak. No scaling-law, robust or first-ever128 claim. If whole or CONT fails, close this width setting without a grid; no trained repair chaining.

## Plan and not done
Explicit launch one candidate, monitor same PID, save full atomic state, restore own best, record whole/default/CONT and actual combination if qualified for adoption. No other folds or delivery changes.

Explicitly launched v337 token128 single candidate; TIMING.json records PID/start, no automatic queue. Active no result yet; retained8 TOTAL80.703440 unchanged.

Live snapshot: PID55704, epoch1, bestPOR77.944334530954 (not final). First observed epoch 14.413s, peak 2599.449MiB, single GPU process at launch; no OOM observed so far, not final peak/speed conclusion. Full atomic state saved, actual initialized score77.926380760724 and330128 trainableparameters. Atlas curate/build completed; no independent disk reload or remaining folds.

V337 final POR78.031174140712, gain+0.104793379988, actualTOTAL80.643913431990; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds394.302705/peakMiB2599.449219. No otherfolds.

Closed v337 width128: natural24/best10 POR78.031174140712, vs335whole-.011038494290/CONT-.055595209042 (CONT41.668915621097). No adoption/adjacentwidth-LRgrid. Predefined42notreached, smallerwork not equal42gain. RAM144665replay/frozenexact,394.303s inclprep/warmup/bestrestore,peak2599.449MiB;23/24 overlap338depth, no isolated speedranking. Keep335 and retained8 TOTAL80.703440. Negative checkpoints not deleted: earlier native cleanup policy rejection still unresolved; do not bypass.

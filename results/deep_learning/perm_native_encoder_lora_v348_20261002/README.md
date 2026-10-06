# V348 PERM native encoder rank4

## Focus and hypothesis
Keep original154 full warm predictive function; jointly update two true expert enc2.expand rank4 modules (3072) and original154 context readout (82720). Zero B initialization, original parameter freeze, native gradient-enabled forward/capture. No trained repair checkpoint, native Gaussian16 and NLL/BCE/jointRPS1 and decoder unchanged. One added PERM phase; 6 phases with original POR/SW, at most8 with retained alternative POR/SW.

## Explored and verification
Prepared; no score yet. Native forward reused from166, but complete154 head restored unlike166. Historical270 warm head-only and late joint both best0/natural14; this is a near negative, not first encoder adaptation. No new head-only job under latest rule. Historical head-only comparison is not concurrent or capacity matched. Original input/split/Q50,80/14/min10,b64,64steps; solo preferred/max2. Initialization and group gradient readback are built into normal training, no separate preflight/review. Report both whole-selected CONT and DEFAULT; LoRA parameter norms at restored best.

## Gain
Unproven. Source154 PERM89.405564856919/CONT74.096911943817; retained336 PERM89.518389398791/CONT74.009637083197; current eight-stage TOTAL80.704705566183 unchanged. Fifth strict80.8 plus useful gain required for other folds; full OOF82 unmet.

## Do not extend
If no whole/CONT improvement, close rank4 enc2/nativewarm setting, no rank/location/LR sweep. 270/162/166/272 near negatives acknowledged; not all backbone adaptation invalid. No repair chain, teachers, labels from validation in training, external voting, or old task restart.

## Timing
Record individual preprocessing/warmup/training/restoration seconds and process wallclock/peak separately. Solo execution, no paired wallclock claims. RAM best replay is not independent diskreload.

TERMINAL STARTUP FAILURE: PID79056 exited, incorrect expand dimension assertion; actual checkpoint128x128x1. No GPU training step/score/checkpoint. Corrected sibling349, all348 source frozen.

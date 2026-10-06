# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# Original model and new seed pair

## Focus and predefined control
User requests joint prediction by retained80.70612972323806 and completed new seed20261010,80.5019511739133. Predefine fixed1:1 weights. First arithmetic mean of final prediction values; then average native predictive distributions and use unchanged official utility decoder. No weight selection, row oracle, cross-head selection or training. Original fixed fifth144665 rows only. Two independently trained8-stage models cost16 successful historical phases in total, not one8-stage model.

## Explored and verification
Arithmetic output averaging computed on actual144665 saved predictions, checked inputSHA and both original official scores on the same truth/order. TOTAL80.42438745976247/POR77.26669834899003/PERM89.5489890469987/SW74.00637653890266. Distribution mixture pending. Native models will be independently reconstructed and loaded from disk; both single-model predictions must match saved native outputs exactly before the mixture result is accepted. Keep each model's native support and common physical decoder actions. Training labels are not read for mixture weights/decoding. Only true held-out labels enter reporting.

## Gain
Arithmetic average vs original -0.28174226347559284 and vs new -0.07756371415082697. This does not settle native distribution fusion. Probability-mixture gain unproven. Strict fifth>80.8/

values; then average native predictive distributions and use unchanged official utility decoder. No weight selection, row oracle, cross-head selection or training. Original fixed fifth144665 rows only. Two independently trained8-stage models cost16 successful historical phases in total, not one8-stage model.

## Explored and verification
Arithmetic output averaging computed on actual144665 saved predictions, checked inputSHA and both original official scores on the same truth/order. TOTAL80.42438745976247/POR77.26669834899003/PERM89.5489890469987/SW74.00637653890266. Distribution mixture pending. Native models will be independently reconstructed and loaded from disk; both single-model predictions must match saved native outputs exactly before the mixture result is accepted. Keep each model's native support and common physical decoder actions. Training labels are not read for mixture weights/decoding. Only true held-out labels enter reporting.

## Gain
Arithmetic average vs original -0.28174226347559284 and vs new -0.07756371415082697. This does not settle native distribution fusion. Probability-mixture gain unproven. Strict fifth>80.8/fullOOF>82 unchanged; no otherfolds and no adoption within single-model stagecap.

## Not worth extending
No weight grid or label-informed combinations in this scoped question. No broad conclusion that ensembles fail from arithmetic averaging. Distribution fusion result must be reported separately; its inference and two-model training cost are explicit.

## Timing and pending
CPU arithmetic scoring0.411s,process/import7.619s. GPU distribution evaluation will be sequential across targets while the existing SOURCE train remains active, cap2. Report per-channel/full comparison wall time, current concurrent train PID, peakGPU memory and prep/warmup included. Pending POR/PERM/SW mixture, final comparison, documentation and atlas.

Terminal fixed1:1 distribution mixture: {"POR": 77.93964020763568, "PERM": 89.56537181009944, "SW": 74.17086138314515, "TOTAL": 80.68957367992631}; vs original -0.016556043312, vsnew +0.187622506013. Both models independently reloaded/native predictions exact on all144665 rows. CONT={"POR": 41.22282838987248, "PERM": 74.01305415036849, "SW": 17.52288214309329}. Channel evaluations sum37.560s/peak1362.866MiB; sequential channels concurrently with existing SOURCE32596, no training interruption/max2. No weight grid or followup launch; original retained/notadopted. Two models8stages each/16historical phases/noaddedtraining. Five-seed study continues; no otherfolds/fullOOF claim. Final artifact FINAL.json.



# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# Original plus new seed distribution weight grid

## Focus and control
User explicitly asks which distribution-fusion weight scores highest, superseding V363 fixed1:1/no-weight-grid scope for this new isolated evaluation only. Compare retained80.70612972323806 and completed seed20261010,80.5019511739133. Use one common original weight w across POR/PERM/SW, new weight1-w, unchanged native probability support and official utility decoder. Predefine coarse0:0.05:1, fine0:0.01:1 (coarse is a subset); one native model forward pass per batch supports all weights, no repeated model training or forward passes for individual ratios.

## Explored and verification
V363 equal probability mixture80.68957367992631, arithmetic output mean80.42438745976247. Both disk-loaded native model predictions reproduce exactly. Current grid pending. Fresh read-only evaluation checks native endpoints and0.5 mixture against those saved artifacts, same144665 row population and coverage once. Store compact compressed action indices and all fixed-weight scores, not large probability caches. GPU max2, serial target evaluation concurrently with existing SOURCE training.

## Gain
Measured best common weight: original0.81/new0.19, TOTAL80.72489290897676, originaldelta+0.018763185738691845. Coarse0.05 best0.85/0.15,TOTAL80.72342481146886. Highest means within tested0.01 grid. Same fifth selects weights, so its ma

ts, same144665 row population and coverage once. Store compact compressed action indices and all fixed-weight scores, not large probability caches. GPU max2, serial target evaluation concurrently with existing SOURCE training.

## Gain
Measured best common weight: original0.81/new0.19, TOTAL80.72489290897676, originaldelta+0.018763185738691845. Coarse0.05 best0.85/0.15,TOTAL80.72342481146886. Highest means within tested0.01 grid. Same fifth selects weights, so its maximum is validation-tuned diagnostic, not independent generalization/fullOOF or proof of a robust gain. POR CONT41.649529086963 and SW CONT17.843954967606 remain below original41.736640291995/17.979399150047; PERM CONT74.019910534281 is above original74.009637083197. Whole-score gain does not preserve every continuous subgroup. No per-target weight oracle, no otherfolds, no current-best replacement automatically. Two independently trained8-stage models,16 prior successful phases total; this is not a single8-stage model.

## Not worth extending
No finer-than0.01 grid, extrapolating/negative weights, row-specific mixtures, learned router or model retraining in this question. Endpoints are part of the grid. If original endpoint remains best, report that the tested pair has no gain; do not claim every ensemble is ineffective.

## Timing and pending
Done: all3 target weight tables, assembled common-weight TOTAL, chosen-grid prediction artifact, CSV and weight curve. Channel evaluations46.256s sum/peak1362.866MiB; study121.382s inclimplementation/gaps/CPUassembly/plot. Load/preparation/warmup/inference/grid included. Current five-seed SOURCE32596 remains active throughout, cap2; timings are concurrent evaluation, not solo inference benchmarks. No finer grid or new training; return to existing five-seed batch. Final evidence FINAL.json and WEIGHT_SCORES.csv.

Terminal101 common weights: best original0.81/new0.19, TOTAL80.724892908977/originaldelta+0.018763185739. Coarse0.05 best0.85/80.723424811469; native0/1/0.5 andsame144665coverage exact. CONT samegridselectedprediction {"POR": 41.64952908696266, "PERM": 74.01991053428122, "SW": 17.84395496760571}. Channel evaluations sum46.256s/peak1362.866MiB, studywall121.382s inclprep/warmup/grid/CPUassembly/implementation/gaps/plot; serial targets concurrent with existing SOURCE32596/max2, notsolo benchmark. Same-fifth validation-tuned maximum within0.01grid only; no finergrid/otherfolds/newtraining/adoption as single8stage model. Two models8each/16 prior phases. Full table WEIGHT_SCORES.csv, curve weight_curve.png, FINAL.json. Five-seed training continues.



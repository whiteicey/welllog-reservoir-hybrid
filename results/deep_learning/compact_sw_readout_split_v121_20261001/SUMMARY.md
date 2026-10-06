# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v121 SW shared versus task-specific final residual readouts
Focus: default gating and conditional numeric density may compete in the same128-dimensional final readout. Add2 identical128-LN-64-GELU-128 bounded residual branches; shared arm sends their mean to both outputs, split sends branch0 to gate and branch1 to density. Shared backbone retained; no claim of fully independent models.
Controls: same original random v104 local SW body/input/prior, same33664 new parameters/equal879577 total trainable parameters, same branch initial tensors and zero residual outputs; original prior function equal. Both use the same separate F.linear gate/density implementation, so per-row computation differences do not define the contrast. Only branch-to-output connectivity changes. Parent-only controlv104 has less capacity; compare mechanism gain against matched shared arm separately from absolute score.
Explored/verified: implemented, no scores yet. Own SW Gaussian NLL+original soft utility1/SAM.1/DropKey.4, original row-uniform sampling/data/Q50/ninecurve206+relative/original fifth/decoder.80/18/min10/b64/64steps/lr3e-4/cos80. Source weights not loaded into SW, only training-side preprocessor/Q dependencies counted. Prior v11 smallTCN/discrete heads tested shared/split encoder scope; this is final residual connectivity on current large Gaussian/local network, not a replay of v11 or evidence 

elivery untouched.
Optional: retain proven connectivity trick only, no hidden extra training phase or external voting.

Live snapshot shared: PID=51456, epoch=6, bestSW=72.235014018835, not final. Actual879577 trainable parameters and initialSW71.197115081472 match planned source function; measured~13.2s/epoch, peak1898.744MiB, single SW process including warmup/load/eval per driver. Split is prepared/untrained, no automatic queue. Same-capacity connectivity comparison remains incomplete; cannot infer gain from current shared snapshot. Current prior5-stageTOTAL80.489472312223 unchanged, no other folds.

V121 shared: restored SW=73.873792847386; original local delta=-0.247959980772; actual fifth TOTAL=80.402686318953;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=34/52, seconds=709.672, peakMiB=1898.744.

Shared arm final: restoredSW73.873792847386, original local model delta=-0.247959980772, actual3-target fifthTOTAL=80.402686318953. best34/stop52,709.672s including load/warmup/train/eval/replay/save, peak1898.744MiB.144665 exact replay/frozen tensors passed. No shared candidate adoption. Keep shared checkpoint as required matched-control evidence until split completes; no automatic queue. Split explicitly launched after shared terminal. Actual serial SW arms, full ablation incomplete; do not call split beneficial/negative yet. Current prior5-stageTOTAL80.489472312223 unchanged, no otherfolds.

V121 split: restored SW=73.849287086711; original local delta=-0.272465741447; actual fifth TOTAL=80.394109302717;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=683.338, peakMiB=1899.771.

Final v121: sharedSW73.873792847386 versus splitSW73.849287086711; same879577 active parameters/init/loss, split-minus-shared=-0.024505760675. Matched connectivity contrast has no positive gain. Both also below original lower-capacity local SW74.121752828158; scope closes without generalizing all task-private heads. Both144665-point replay/frozen checks passed. Actual3-channel fifthTOTAL shared=80.402686318953, split=80.394109302717; retain prior5-stageTOTAL80.489472312223, strict808false, no otherfolds. Pair actual serial wall=1462.846s including gap=61.502s; singles709.672/683.338s, peak1898.744/1899.771MiB. Split briefly overlapped0.397s runtime-only TF32 probe, no full training precision change. Negative best/last removed with manifest; all source/current positives and delivery retained. Next work changes mechanism, no additional split-head width/bound sweep. FullOOF>82 still unmet.



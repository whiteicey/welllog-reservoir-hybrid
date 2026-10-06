# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v242 final FF and distribution coadapt

## Focus
Retain original104 structure/function and probability mass correction. Permit only final existing global FF representation to learn jointly with mass readout; lowFFlr3e-5/readout1e-4. No new memory, no trained211/219/235 weights. This is one SW stage.

## Explored and validation
Prepared/untrained. Control211 mass-only keeps alloldweights frozen; same binNLL/BCE/SAM, seed and budget. Differentiable finite-edge CDF needed for backbone gradients; report numerical initial score separately.

## Gain
Unproved. Original validation144665, ownSW checkpoint and actualcombinedscore. Individual/batchwallclock, max2 and peakmemory recorded after real training.

## Not worth continuing
Earlier newmemory+mass235 yielded only+0.00823 below219. Do not extend that memory chain.242 checks existing positive representation coadapt rather than more tiny frozen-only repair widths. No current weights replaced.

V242 joint final SW74.121752689769, actualTOTAL80.612475369558; learned gain+0.000000000000, vs211-0.025869932858. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

242 final no learned gain,best0/stop13/earlyterminated,SW74.121752689769,214.8334s/976.59MiB; actualassembly complete. Delete redundant best/last checkpoints with manifest; close this finalFF lowLR+binNLL coadapt, not allbackbone optimizatio

# v242 final FF and distribution coadapt

## Focus
Retain original104 structure/function and probability mass correction. Permit only final existing global FF representation to learn jointly with mass readout; lowFFlr3e-5/readout1e-4. No new memory, no trained211/219/235 weights. This is one SW stage.

## Explored and validation
Prepared/untrained. Control211 mass-only keeps alloldweights frozen; same binNLL/BCE/SAM, seed and budget. Differentiable finite-edge CDF needed for backbone gradients; report numerical initial score separately.

## Gain
Unproved. Original validation144665, ownSW checkpoint and actualcombinedscore. Individual/batchwallclock, max2 and peakmemory recorded after real training.

## Not worth continuing
Earlier newmemory+mass235 yielded only+0.00823 below219. Do not extend that memory chain.242 checks existing positive representation coadapt rather than more tiny frozen-only repair widths. No current weights replaced.

V242 joint final SW74.121752689769, actualTOTAL80.612475369558; learned gain+0.000000000000, vs211-0.025869932858. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

242 final no learned gain,best0/stop13/earlyterminated,SW74.121752689769,214.8334s/976.59MiB; actualassembly complete. Delete redundant best/last checkpoints with manifest; close this finalFF lowLR+binNLL coadapt, not allbackbone optimization.243 SWdecisionrank and244 PERMconditionalCDF now live max2; oneaddedstageeach. Bestoptional7TOTAL80.624489319708/default5 intact; gate808/fullOOF82 unmet.



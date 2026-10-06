# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW joint memory and mass readout in one stage

Focus: combine evidenced memory232 and mass211 architectures through one fresh training phase from104,not sequential trained repairs.14304new tokenmemory +12576mass head; both zero outputs,old104weights frozen. Masspatch also receives differentiable memory-updated hidden features.

Explored: prepared/untrained,owncategoricalNLL+BCE/utility0/SAM0.1/LR1e-4,originalfold5/Q50/80-14/1024input/256bins/800actions. Finite255-edge GaussianCDF gradients with constant0/1endpoints prevent infiniteedge derivativeNaNs; component-bin differences weighted exactly as original. Train only scored mask,checkpoint1024pointCDFchunks.

Gain: unproved,controls211SW74.147622622627,21974.148179431612,232 provisional74.133475000603. No trained211/219/232 weights. OneSWextra phase/six total,optional7 withPERM230. Gate function may change through newmemory; loss differs from232continuous+utility1,compound trial not pure attribution.

Not worth continuing: standalonememory belowmass control, width128 below32. This tests coupling through learned features with same-stage budget rather than adding another stage.

Timing includes loading/warmup/eval,actualoverlap/peak. Actualmask finitegrad training and restoredRAM/frozen/144665 assembly required; no numericalpreflight/review. No otherfolds until strict80.8+gain.

V235 joint final SW74.129980102971, actualTOTAL80

# SW joint memory and mass readout in one stage

Focus: combine evidenced memory232 and mass211 architectures through one fresh training phase from104,not sequential trained repairs.14304new tokenmemory +12576mass head; both zero outputs,old104weights frozen. Masspatch also receives differentiable memory-updated hidden features.

Explored: prepared/untrained,owncategoricalNLL+BCE/utility0/SAM0.1/LR1e-4,originalfold5/Q50/80-14/1024input/256bins/800actions. Finite255-edge GaussianCDF gradients with constant0/1endpoints prevent infiniteedge derivativeNaNs; component-bin differences weighted exactly as original. Train only scored mask,checkpoint1024pointCDFchunks.

Gain: unproved,controls211SW74.147622622627,21974.148179431612,232 provisional74.133475000603. No trained211/219/232 weights. OneSWextra phase/six total,optional7 withPERM230. Gate function may change through newmemory; loss differs from232continuous+utility1,compound trial not pure attribution.

Not worth continuing: standalonememory belowmass control, width128 below32. This tests coupling through learned features with same-stage budget rather than adding another stage.

Timing includes loading/warmup/eval,actualoverlap/peak. Actualmask finitegrad training and restoredRAM/frozen/144665 assembly required; no numericalpreflight/review. No otherfolds until strict80.8+gain.

V235 joint final SW74.129980102971, actualTOTAL80.615354964179; learned gain+0.008227413202, vs211-0.017642519656. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.



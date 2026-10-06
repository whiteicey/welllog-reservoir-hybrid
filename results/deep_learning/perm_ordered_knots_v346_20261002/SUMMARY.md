# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM ordered numerical knots v346

Focus: retain fresh336 rational-body/readout and low-rank cross-feature code, add zero numerical residual before stem GELU; jointly train all new encoder/shape/cross/numeric/readout in one PERM phase. Only original154 trained checkpoint loads, no336/312/276 weights.

Hypothesis/control: nine originalcurve train_raw inputanchors of64trainingwells fit q05/q95, fullrange fallback if robust range too small, remove constantcurve when fullrange small. No validationinput/label/Q in boundary fit. Eight initiallyuniform cumulativePLE components peractivecurve; widths=span*(1/512+(1-8/512)*softmax(logits)), fixed endpoints/ordered/minwidth. Scalar207/missing/relative bypass unchanged, basis saturates outside boundaries. ZeroLinearS->48/no bias +.25tanh residual beforeGELU, after it inherited336crosshook remains. Learnedwidths vssameuniformfixed control, fixed widths nottrainable; +S<=72 trainableparameters acknowledged/no dummy weights. Projection<=3456, learnedextra<=3528/active<=40914, fixedactive<=40842; actualconstantcurve count read in ordinarytraining.

Explored/verification: research primary2026paper/officialcode and exact?history276localPLEpositive butnotretained, POR305/332 andoldhatcalibration negatives documented in ../cross_domain_research_20261001/ADAPTIVE_DEPTH_AND_ORDERED_NUMERIC_RESEARCH_20261002.md. This is notfirstPLE, no guarantee i

proof. Original9/206/split/Q50/seed20261010/80pat14/min10/b64/64steps/cos80/max2. Explicitsolo preferred measured contention, noqueue; after345pair asavailable. Report individual/pairwall withserialgaps, prep/warmup/restoration/peak. No teacher/extra labels/pseudo/oracle.

2026-10-02 learned explicitsolePID50464after345pairterminal/globalGPU0; ordinary INITIALPERM89.405566626894 actual40914params, epoch1normal/914.309MiB. Inputboundsfit9columns then64trainanchors avoids206wide temporary; sameinputpopulation. Fixedarmnotlaunched/noautoqueue; gainunproven. SPO+ alternative research completed, notimplemented/noGPUqueue.

V346 final PERM89.506387553016, gain+0.100822696097, actualTOTAL80.647763361628; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds153.535463/peakMiB914.309082. No otherfolds.

Learned natural19/best5 PERM89.506387553016/CONT74.005935936341/DEFAULT97.478168914750; vs336 whole-0.012001845775/CONT-0.003701146856, native154CONT-0.090976007476. Actualknots logitnorm0.760082901/projection2.225332022, intervalfractionmin0.107098334/max0.142446488, activeS72; learning occurred, no causalbenefitclaim.153.535463s/914.309MiBsolo incltraininputfit/prep/warmup/restoration,144665RAMbestreplay/frozenexact notdiskreload. Noadoption/fixedpending. Fixed explicitPID56572afterlearnedterminal/globalGPU0; noautoqueue.

V346 final PERM89.506393762633, gain+0.100828905714, actualTOTAL80.647765534994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds151.729385/peakMiB903.087891. No otherfolds.

## Terminal paired conclusion
Bothnatural19/best5, learnedPERM89.506387553016/CONT74.005935936341 vsfixed89.506393762633/CONT74.006486486651; learned-fixedwhole-.000006209617/CONT-.000550550310. Vs336learnedwhole-.012001845775/CONT-.003701146856, fixed-.011995636158/-.003150596546 => no adoption/no knotsgain, close current8uniform orderedknot/stemsetting, no K/bounds/family/LRgrid. Not allnumericembeddinginvalid. Learnedlogits changednorm.760083/projection2.225332/widthfractions.107098-.142446, no minfloorcollapse; fixedzero logits/equal.125 verified. Learned40914 vsfixed40842/72trainablecapacitydifference disclosed. Individual153.535463/151.729385s solo includestraininputfit/prep/warmup/restoration; pair398.630086s includes explicitserialgap/stagger241.804, peaks914.309/903.088MiB.144665RAMbestreplay/frozenexact notdiskreload/fullOOF/single seed,19prefixsamework. Current341TOTAL80.704705566183 retained/strict808fullOOF82unmet/nootherfolds.



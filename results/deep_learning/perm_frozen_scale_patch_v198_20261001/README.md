# v198 Freeze learned154; one small scale calibration patch

## Focus
Keep every proven154/SOURCE parameter and function component. Train only4656 newparameters reading frozen learned128-dimensional head features, adding boundedraw-scale corrections. No gate/location/logweight/body update; original weights remain unchanged. This is an alternative one-stage repair from154, never stacked on197. Whole6 stages under max8; added head counts POR0/PERM1/SW0.

## Explored and validation
Prepared/untrained. Exact154SHA2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11 loaded/frozen. Capture originalcontext_head.output input (learned128 features), Conv128to32/GELU/Conv32to16zero. Newrawscale delta0.1*tanh(u/0.1), preserves initial derivative1 and boundedrange; standarddeviation remainssoftplus(originalrawscale+delta)+0.025. Original full learned head predicts means/logweights/gate unchanged, Gaussian PDF/CDF and canonical256grid/800action decoder synchronized. Default/continuous NLL+BCE+jointRPS1 unchanged, onlynewpatchlr1e-4/AdamW/noSAM/80-14/min10/b64/64steps. Source feature/density forward no_grad; frozenlearnedhead remains as reference, no teacher loss. No addedCRPS/localcurve patch. Original9curves206+63amp/seed/train64/fold5/Q50 unchanged.
Related159 allowed scales while simultaneously fitting a SOURCE-initialized larger48output head and failed (best0), with bound0.5. Current trial preserves trained154 and isolates a conservative scale-only repair; this is not evidence that generic scale learning works. New optimizer, no historical trajectory resumption. Atomicstate/frozen checks/all144665 restored ownbest and actualPOR151/SW104 assembly at final. Initialpredictive function must match154; no fabricatedTOTAL.

## Gains
No training/no gain. Functional control154PERM89.405564856919/TOTAL80.612475417994. Report initial equality, learned gain and actualTOTAL separately. No new-stage stacking; eachhead maxone addition. Strict80.8+clear gain beforeotherfolds; fullOOF>82 unmet.

## Not worth continuing
No evidence yet. If no gain, close this bounded0.1scale-only repair; no scale/LR/widthgrid or generalized rejection of calibrated distributions. Preserve original154 evenif repair fails. Timing singletraining/actualpairedwall/data/initial/finaleval/peak/warmup disclosed, not full6phase time.

V198 scale final PERM=89.405831181353, compactcontrolgain=+0.027436970209; actualTOTAL=80.612568631546, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=126.469, peakMiB=786.465; all144665replay/frozen exact.

Verified current research best six-stage TOTAL80.612568631546, versus five-stage154+0.000093213552, PERM+0.000266324434. Tiny singlefold/singleseed gain, not significance or promotion; best1/stop15/126.468553s/786.465MiB, all144665 actualassembly/reload/frozenverified. New198CURRENT_RECIPE preserves old154/deliveries; keep positive198best/last. Source-counted6, addedhead POR0/PERM1/SW0, max8. FuturePERM candidates siblings from154 only, not198warmstack.

# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v195 One additive analytic-CRPS term on proven PERM structure

## Focus
Follow latest user direction: retain acquired mechanisms, modify one small part. Keep the entire effective154 SOURCE-frozen Gaussian location/logweight readout, same128trunk/82720active params, NLL+BCE+jointRPS1. Only add0.1 analytic continuous Gaussian-mixture CRPS, nondefault rows. No replacement readout/backbone or added module/parameter.
Mathematical identity CRPS=E|X-y|-0.5E|X-Xprime|, closed Gaussian absolute expectation. Source: https://github.com/properscoring/properscoring/blob/master/properscoring/_crps.py (Gaussian closed form and expectation identity); nativePyTorch implementation, no library dependency. The 16Gaussian mixture pair expansion is our analytic application, not sampled ensemble CRPS.

## Explored and validation
Prepared/untrained. Existing finite255edge jointRPS remains; analytic conditionalCRPS additionally covers whole standardizedlog10 support and both crosscomponent interactions. No decoder/grid/action/window change. Default rows zero for extra term, same global/Q weighting; no gate/scale/body updates. Same SOURCE initialization, seed, architecture, originalGaussian output, LR3e-4/AdamW/noSAM/80-14/min10/b64/64steps. Matched current-budget control165adamw best0/14 PERM89.344760174753; historical154 trained80/18 best31/49 PERM89.405564856919 is absolute reference with different

est31/49 PERM89.405564856919 is absolute reference with different patience, not same-budget ablation. Historical154 first gain was epoch16 following15bad; new14 budget does not prove convergence if both stop earlier.
Retaining proven architecture/loss differs from retaining all learned154 weights: loading them then training another stage would count sixth stage. This candidate starts sameSOURCE and replaces existing PERM phase; original positive154 weights never modified. Whole recipe5 inclSOURCE/POR113/POR151/PERM195/SW104. Atomic last/official own best/frozen/exact144665 replay and actual fixedPOR151/SW104 assembly at final. No external labels, teacher or target statistics from validation.

## Gains
No training yet; no gain. Require own initial gain and fair165adamw delta, absolute154 gap and actualTOTAL separately. Existing compact best80.612475417994 intact, strict80.8+gain before other folds; OOF>82 unmet.

## Not worth continuing
No evidence yet. If addedCRPS fails, close this0.1 conditional analytic augmentation; do not scan coefficient/family or erase existing jointRPS benefit. Proper score does not imply officialargmax gain. Exact formula evaluates256component pairs per row, checkpoint chunks1024; report actualtime/peak/overlap, not identical compute/full5phase cost.

v195 explicit launchPID87044, normalGPU guard only193otherjob, temporarySYSTEM_REQUIREDTrue. Same82720 params/originalGaussian output/NLL/BCE/jointRPS; only analyticCRPS0.1 added, no new module or backbone replacement. InitialPERM89.3447601747529 exactly sameSOURCE/control165adamw score; learned gain not shown yet. Snapshot {'status': 'running', 'pid': 87044, 'stage': 'perm_conditional', 'epoch': 3, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 3, 'peak_MiB': 786.46533203125}, original154 positive weights/delivery unchanged; five phases and no otherfolds.

V195 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=211.378, peakMiB=786.465; all144665replay/frozen exact.

FINAL v195 PERM89.344760174753 best0/stop14, learnedgain0, matched165AdamW delta0; actualTOTAL80.591193779236,144665 exact replay/frozen. Added analyticCRPS0.1 no gain at80-14; original jointRPS benefit154 retained, no coefficient grid. Training211.377959s/peak786.465332MiB, actualstaggered193/195 overlap, data/initial/final eval included, standaloneassembly excluded/no separatewarmup; not full5phase/equalcompute cost. Negative best/last eligiblecleanup with SHA; source/config/evidence preserved.



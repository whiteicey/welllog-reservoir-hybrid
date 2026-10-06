# v193 SOURCE-referenced adaptive mass transport

## Focus
Borrow AdaCat adaptive interval widths and probability masses, applied to the frozen SOURCE conditional CDF coordinate u=F0(z). This is a task-specific bounded probability transport, not exact AdaCat target-space discretization.
Sources: https://arxiv.org/abs/2208.02246 and https://colinqiyangli.github.io/adacat/

## Explored and validation
Prepared/untrained. Same128 context trunk/82720 active parameters/32 outputs as Gaussian174. First16 outputs widths, last16 masses: 0.001+0.984*softmax(tanh(output)), each sums1. Their cumulative nodes define monotone piecewise-linear T(u); newCDF=T(F0(z)), density=f0(z)*T'(F0(z)). Strictly positive widths/masses and bounded logits prevent arbitrarily narrow mass intervals. Uniform widths/masses at zero output give algebraic identity; actual initial predictions still must be recorded. Unlike191IQF, no SOURCE quantile inversion, replacement exponential tails or interpolation approximation. Unlike POR150 smooth CDF warp, this is PERM adaptive mass and width transport; historical relation does not guarantee gain. Both source features and original raw densities frozen; no teacher-label loss, no trained174/154 warmstart. SOURCE feature cache only within the two SAM passes, cleared each optimization step/eval.
NLL/BCE plus conditionalRPS1/finiteutility0.1, original255 edges/256mass/800actions, Q50, SAM0.05/lr3e-4/80-14/b64/64steps unchanged. Own labels only. Proper normalized continuous density includes exact transport Jacobian; original standardized-log10 Gaussian NLL and parameter-independent physical Jacobian convention unchanged. Density jumps at adaptive boundaries are allowed. Full144665 evaluator/own checkpoint restore/frozen proof and actual POR151/SW104 assembly only; no fabricatedTOTAL.

## Gains
Not trained, no gain. Primary representation comparison Gaussian17489.364985921273 has same capacity/source/trunk/loss/SAM/budget, different parameterization/computation; require initial score and learned improvement separately. Absolute15489.405564856919/TOTAL80.612475417994 preserved. Source original89.344760174753. Five total stages inclSOURCE; no otherfolds unless strict80.8+gain.

## Not worth continuing
No evidence yet; if no learned gain, close this16interval bounded adaptive transport, no interval/floor/strength grid or class-wide conclusion. Timing single training versus actual paired wall, data/initial/final eval/peak/overlap must be recorded; not full5phase cost.

v193 adaptive explicitly launchedPID77412, normalGPU guard found only192multiply, temporarySYSTEM_REQUIREDTrue. Active82720. InitialPERM89.344788677861, minusSOURCE+0.000028503108; algebraic uniform identity does not yield exactly identical measured decoding, no initial difference counted as learning. Possible finite-precision/readout implementation variation is inference, not proven cause. Same-SAM-step cache cleared eachstep/eval. Source/config hashes now frozen; no automatic retries. Snapshot {'status': 'running', 'pid': 77412, 'stage': 'adaptive', 'epoch': 1, 'best': 89.34478867786068, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}, no final mechanism gain.

V193 adaptive final PERM=89.344788677861, compactcontrolgain=-0.033605533283; actualTOTAL=80.591203755324, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=342.805, peakMiB=786.465; all144665replay/frozen exact.

FINAL v193 PERM89.344788677861, best0/stop14, learned gain0; control delta-0.020197243412, actual TOTAL80.591203755324,144665 exact replay/frozen realassembly. Close this bounded adaptive transport only, not allAdaCat. Training342.804890s/peak786.465332MiB, actualstaggered193/195 overlap, data/initial/final eval included, standaloneassembly excluded/no separatewarmup; not full5phase/equalcompute cost. Negative best/last eligiblecleanup with SHA; source/config/evidence preserved.

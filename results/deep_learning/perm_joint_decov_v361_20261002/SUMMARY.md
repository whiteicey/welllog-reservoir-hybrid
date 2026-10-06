# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V361 joint PERM channel covariance regularization

## Focus
Keep all fresh336 positive architecture and original154 frozen prior. Train all37386 fresh encoder/rational/cross/readout parameters jointly in one additional PERM phase; no extra learnable parameters. DeCov-inspired scale-normalized auxiliary on actual final48 readout features; not full original DeCov reproduction. A different training constraint after finite depth/path repair failed, no wholesale replacement.

## Explored and verification
Prepared source compiled; no numerical preflight/review/GPU training yet. Two predefined decorrelation/identity arms, identical architecture/initialization/main objective/sampling and diagnostics, auxiliary .01/3 vs0. Training CONT core only: original row%32, dedup/hash,max8perwell,max256,min8. FP32 C=centeredH.T@centeredH/N; aux=sum(offdiag(C)^2)/(trace(C)^2+1e-8). Denominator and H retain gradients; main loss and Q50 unchanged. No target distance/label pair/prediction feedback/teacher/pseudo/validation labels. Covariance uses unweighted selected training features; Q stays in the main objective. Identity calculates same diagnostics with coefficient0, no RNG changes. Evaluation unchanged.

## Gain
Unproven. Reference PERM33689.51838939879066/CONT74.00963708319706 and TOTAL80.70612972323806. max80/min10/pat10 until strict frozen whole SOTA then14, b64x64/cos80/own loss/original spl

). Denominator and H retain gradients; main loss and Q50 unchanged. No target distance/label pair/prediction feedback/teacher/pseudo/validation labels. Covariance uses unweighted selected training features; Q stays in the main objective. Identity calculates same diagnostics with coefficient0, no RNG changes. Evaluation unchanged.

## Gain
Unproven. Reference PERM33689.51838939879066/CONT74.00963708319706 and TOTAL80.70612972323806. max80/min10/pat10 until strict frozen whole SOTA then14, b64x64/cos80/own loss/original split/seed/9curves206 contract. Whole checkpoint selection only; CONT at same restored checkpoint. Report actual pair delta AND absolute reference delta, no single-seed robustness/generalization claim. Default5/current8 unchanged; recipe6 with default other heads or8 with341/351. Strict fifth TOTAL>80.8+useful gain before remaining folds/fullOOF>82 unmet.

## Not worth extending
Basis rotations may lower regularizer without useful behavior; physical correlations may be useful; epsilon can encourage low variance collapse. Log selectedN/trace/offdiag ratio/main andaux losses from normal training only. If no useful whole+CONT pair benefit, close this coefficient/layer, no variance/contrastive/orthogonal/coeff grid. Negative does not disprove all representation regularization. Positive historical336code only, no336/358 trained repairs or extra chain.

## Timing and next
Explicit finite serial arms, no automatic queue; prefer solo GPU due measured contention/max2. Record individual prep/warmup/restoration and peak, paired wall including gaps. Atomic full laststate/explicit resume only, RAMbest replay is not independent disk reload. Research: [primary-source audit](../cross_domain_research_20261001/PERM_JOINT_DECOV_RESEARCH_20261002.md).

V361 decorrelation explicitsolo launch PID604 after360terminal/globalGPU0; identity preparednotlaunched/noqueue. Sourcecompiled/ordinaryinitialevaluation and training pending actual readback; no gain yet.

V361 decorrelation terminal PERM89.51434795830305/CONT73.997176916398, vs336whole-0.0040414404876116805/CONT-0.012460166799058925, best/stop5/15, effectivepatience10/SOTAbrokenFalse. Actualdiagnostic8TOTAL80.70471521906741; notadopted beforepaireddecision. 116.3372768999543s solo/897.4375MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.

User switches to five complete seeds of80.706130 eight-stage recipe. V361 decorrelation terminal; identity remains untrained/held, no paired gain or broad negative claim. Preserved results/currentrecipe.



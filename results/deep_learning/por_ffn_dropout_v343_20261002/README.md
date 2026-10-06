# POR FFN dropout v343

Focus: user-requested dropout. Preserve v341 positive architecture and conditional-RPS code; add p=0.1 after GELU in each of its two width64/FFN128 curve blocks. No attention dropout, no new parameters, no added training phase beyond one POR augmentation. Fresh encoder, temporal/curve interaction and readout jointly train; original151 checkpoint only. Never load trained341/338 repair weights.

Hypothesis/control: regularization improves whole POR and continuous-value reading. Historical v341 p0: whole78.04643071793232, CONT41.736640291995144, natural42/best28,134752 parameters. Parameter names/shapes and initialization preserved; dropout RNG changes training trajectory. Historical single-seed comparison is not robustness proof. Predefine best-through42 when reached; later improvements are not equal-work evidence. Whole-target official best selection, not continuous-only epoch selection.

Explored/verification: source prepared, syntax checked only; ordinary trainer checks initial POR151 score, gradients, frozen state and restores/replays best. No extra numerical preflight or independent review. Original nine curves, train-only Q50, fixed fifth, seed20261010, 80/14/min10/b64/64steps, max2 GPU. Prefer solo due observed same-GPU contention. Atomic resumable state, explicit resume only.

Gain: not trained yet; unproven. Adoption requires terminal whole and CONT improvement over341, then actual fixed-column combination. Strict fifth TOTAL>80.8 and complete grouped OOF>82 unmet; no otherfold launch.

Do not pursue: no probability/LR/width grid or checkpoint chaining. If this setting is negative, close this setting rather than claiming all dropout fails. Prepared342 held while fulfilling dropout request.

Timing: report individual elapsed includes preprocessing/warmup/restoration and process wall time, serial/concurrent state, peak memory. No benchmark ranking from overlapping runs.

Explicit solo launch PID4160 after341terminal/globalGPU0. Observedinitial77.926380760724/134752 params and epoch1 POR77.944651857837, peak2229.881MiB; running snapshot not finalgain.

V343 final POR78.044847767677, gain+0.118467006953, actualTOTAL80.648015520080; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds487.441191/peakMiB2229.881348. No otherfolds.

## Terminal conclusion
42 epochs/best28, POR78.044847767677, delta341-0.001582950256; CONT41.732088381788, delta-0.004551910207; DEFAULT unchanged97.405597888869. Actual144665 columns343POR/336PERM/326SW TOTAL80.704230681106, delta retained-0.000474885077, diagnostic only/not adopted. Same historical341 parameter init/capacity/loss and actual42 budget; single seed, stochastic trajectories differ, no robustness claim.487.441191s individual/491.917613s process wall solo includes prep/warmup/restoration,2229.881MiB peak. RAMbest replay/frozenexact, not independent disk reload. Close this FFN p.1 setting, no dropout grid. Retain341 optional8 TOTAL80.704705566183/default5 and prior deliveries. No otherfolds/fullOOF82 unmet. No new GPU launch.

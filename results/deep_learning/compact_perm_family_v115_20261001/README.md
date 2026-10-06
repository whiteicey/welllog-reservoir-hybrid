# v115 compact PERM heavy-tail family comparison
Focus: test whether fixed Gaussian tails limit PERM compact readout; compare t3 and Cauchy with source Gaussian component IQR matched. Raw source parameters identical; initial predictive functions differ, so initial family change is not training gain.
Controls: retained v100 Gaussian PERM43431 active parameters, same source/init/data/Q50/optimizers/80-18/min10/bs64/64steps. Two fixed family arms; own PERM proper likelihood/CDF synchronized, original grid/action scoring unchanged. Original gates/routes/side train; all source density rows and backbone frozen. No external voting, test labels or row-wise selection.
Explored/verified: implementation prepared, no training score yet. Source-only initialization for new PERM; fixed POR=v114, SW=v104. Five total dependency stages including source and both POR phases. Two explicit GPU tasks maximum, atomic complete last states and explicit resume.
Gain: not established; Gaussian89.378394211144, prior actual compact TOTAL80.489472312223. Report own-family learning gain and matched-budget Gaussian difference separately. Cauchy has no finite mean/variance.
Not worth continuing: no evidence yet for this scoped family change. Prior failed warm PERM mean scopes and POR width scope are not reopened.
Pending: direct training, restored-best/population checks and actual three-target assembly; strict fifth>80.8 plus gain before remaining folds. Full originalOOF>82 remains unmet. Prior delivery preserved.
Optional: only evidence-based refinement with stage budget, no automatic queue.

V115 t3: restored PERM=88.396923292819; Gaussian delta=-0.981470918325; actual fifth TOTAL=80.145957490810;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=2/20, seconds=131.115, peakMiB=792.793.

V115 cauchy: restored PERM=88.391510248652; Gaussian delta=-0.986883962492; actual fifth TOTAL=80.144062925351;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=9/27, seconds=164.107, peakMiB=786.465.

Final decision: no new family adopted. PERM t3-Gaussian=-0.981470918325, Cauchy-Gaussian=-0.986883962492. Both final best checkpoints reproduce all144665 predictions and frozen parameters exactly. Initial family shifts caused ~1point loss not recovered by gate/routes adaptation. Close this fixed-IQR/frozen-density family scope; no general claim against all heavy-tail models. Current5-stage fifthTOTAL=80.489472312223, unchanged, strict808 false; original fullOOF>82 still unmet. Concurrent pair wall=169.044s; per-run/peak in pair_timing.json, no equal-work serial speedup evidence. Negative best/last checkpoints removed with manifest; source/current/positive controls and old delivery preserved.

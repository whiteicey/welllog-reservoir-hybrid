# Current eight-stage continuous-value bottleneck diagnosis

Fixed original fifth, 144665 rows. Validation-label analysis only; it must not define per-row models, weights or train features. No new training phase, no oracle selections, no test labels. Results describe this selected recipe and do not prove backbone causality or generalization.

|Target|Continuous rows|Continuous score|Continuous zero utility|Default score|Whole gain vs default5|
|---|---:|---:|---:|---:|---:|
|POR|50308|41.724511|40.83%|97.405598|+0.115832|
|PERM|49132|74.009637|6.63%|97.494440|+0.112825|
|SW|43879|17.963442|66.28%|98.639881|+0.047790|

Current TOTAL 80.703440141304; reaching80.8 needs 0.096559858696 more TOTAL and the gate requires strictly above it.

With all other scores fixed, the required continuous-stratum gain for each target is:
- POR: +0.925554 CONT points (whole-target +0.321866). Arithmetic opportunity only, not a feasible or promised gain.
- PERM: +0.812321 CONT points (whole-target +0.275885). Arithmetic opportunity only, not a feasible or promised gain.
- SW: +0.909568 CONT points (whole-target +0.275885). Arithmetic opportunity only, not a feasible or promised gain.

The low continuous score alone cannot establish a weak backbone: original official tolerance, default/continuous population, ambiguity across wells, representation and likelihood-to-decision readout all contribute. New hypotheses must change encoder and readout jointly and be tested from original allowed sources, never chain retained trained repairs. Preserve DEFAULT performance when measuring CONT improvements. No new architecture is selected by this diagnostic; no row-wise or per-well oracle predictions created.

Completed: baseline/retained whole and strata comparison. Not done: held-out full five-fold proof, causal backbone-versus-readout attribution, further training candidate. Closure: do not repeatedly recompute unchanged diagnostics or treat validation near-miss distribution as train labels.

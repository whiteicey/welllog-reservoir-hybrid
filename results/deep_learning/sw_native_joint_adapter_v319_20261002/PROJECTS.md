# V319 native SW joint backbone / readout adaptation

Focus: retain original104 native Gaussian mixture and jointly train its whole backbone, condition, gate and density. Control:845913 parameters; adapter:852185 (+6272 fresh zero-output residual FF code from225). Original weights loaded strictly before wrapping; no old repair weights or prior categorical blending.

Hypothesis: state and continuous readout need backbone coadaptation. Both use native NLL/BCE+utility1, SAM0.1, training DropKey0.4, same original initialization/data/Q50/fifth split/seed. Original parameters lr3e-5; new adapter3e-4. Not capacity matched; additional parameters alter global SAM norm, so paired delta includes that coupling. Only last global FF receives bounded0.1 residual. This is not first use of225 adapter;225 frozen body had +0.00703658 SW.251/260 continuous-only full-body attempts were negative.

Validation: direct training, initial official SW must match74.121752828 within1e-6; record gradients, restore own official whole-SW best and replay144665 rows. Independent disk reload only if promotion/final delivery. No full OOF or remaining folds launched. Max80/patience14/min10/64x64, max2 GPU. One added SW phase,6 total standalone; ?8 with existing other heads.

Results: prepared, no score yet. Gain unproven. Do not adopt default-only micropositive at expense of continuous representation. If no learned/continuous useful gain, close this intersection without layer/rank/width/LR grids. No teacher/distillation/external data.

Timing: individual wall clock includes preparation and initial/final evaluation; record concurrent pair wall clock separately, GPU peak measured. Concurrency is not isolated speed evidence.

Done: source/trainer/plan prepared. Next: train control/adapter, actual paired decomposition and assembly. Unfinished: qualification >80.8, full original OOF>82. Later optional: only evidence-backed continuation.

LIVE LAUNCH verified: control PID440 and adapter PID76864, both initialSW74.12175282815758 exact source replay. Control845913/adapter852185 all trainable, initial training peak1722.19/1723.05MiB. Two GPU jobs; snapshots are not final gain. Atlas curated/built:1167 experiments,231 dead ends;318 closure recorded.

V319 control: SW74.121752828158, actualTOTAL80.612475417994; native104 whole backbone/state/Gaussian density jointly trainable, with or without fresh6272 FF adapter, oneaddedSWphase. No fullOOF/otherfolds.

V319 adapter: SW74.121752828158, actualTOTAL80.612475417994; native104 whole backbone/state/Gaussian density jointly trainable, with or without fresh6272 FF adapter, oneaddedSWphase. No fullOOF/otherfolds.

FINAL319 CLOSED: control best0/earlystop11, adapter best0/earlystop12; each SW74.12175282815758, actual6TOTAL80.61247541799393. Paired whole/CONT0; selected sourceweight deltas0 because initialization selected, live body/state/density updates verified nonzero. CONT17.63188906048012 is original source, not learned gain. No adoption/LR-layer-grid/fullOOF/otherfolds; no independent diskreload. Individual363.678/379.329s, peak1722.193/1723.052MiB; concurrentpair405.489s/stagger21.527s inclprocess-to-lastterminal, excludeprelaunch/CPUfinalization. Default5/retainedoptional8 unchanged.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

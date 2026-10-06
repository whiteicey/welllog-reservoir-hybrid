# Goal
Improve fixedfifth whole+CONT/fullOOF82 unmet.
Done: prepared256 numeric basis residual joint encoder/readout.
Next: launch onlyafter freeGPU, actual results/pairedfinal/docs.
Not done: train/gain/folds/promotion. Noautomatic followups.

Basis constant protection (before launch): q95-q05<1e-6 falls back to training-input full min/max; if full range remains<1e-6 disable that curve residual. Spacingfloor1e-3 remains fixed, originallinear path retained. No validation/label statistics. [Primary/source and historical research](../cross_domain_research_20261001/POR_NUMERIC_BASIS_RESEARCH_20261002.md).

332 LIVE: rbf PID76224/cumulative PID42448 launched after331 terminal, globalmax2. OrdinaryinitialPOR77.92638076072386/76848trainable both, no learned gain yet. Only151 trainedcheckpoint +fresh302code+256 residual numeric basis; source/config frozen once launched, allnewbodyattention/readout joint, onePORphase/max8. Attention warn-only nondeterminism as historic302; single-seed tiny delta not robustness proof. Train-onlyq fit included in timing.

V332 final POR78.020299603815, gain+0.093918843091, actualTOTAL80.640651070921; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds515.142976/peakMiB1819.070801. No otherfolds.

V332 final POR78.022463922177, gain+0.096083161454, actualTOTAL80.641300366430; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds513.957182/peakMiB1819.070801. No otherfolds.

## Final closure
Both naturalstop20/best6. RBF POR78.02029960381482/CONT41.60782861942168; cumulative78.0224639221774/CONT41.618027814697186. RBF minus cumulative whole -0.002164318363/CONT -0.010199195276. Bothbelow302: whole -0.010253190330/-0.008088871967, CONT -0.067251287650/-0.057052092374. Projectionnorms0.782424/0.712187 confirm nonzerolearned numeric residual, not dead initialization. No adoption or bandwidth/count/LR scan; close this numeric-token setting. Equal nominal80-14 but historical302 ran longer, no equalwork causal claim; CPUfinalizer inherited predefined27 fields are null/notapplicable (332trainer neverpredeclared27). Best144665 RAM replay/frozenexact, not independentdiskreload. Pairwall520.725597s/stagger1.166s, individual515.142976/513.957182s, peaks1819.071MiB; inputstatistics/prep/initial/train/restore included, implementation/finalization excluded. Retained8TOTAL80.699037441/default5 unchanged/strict808false/fullOOF82 unmet/nootherfolds.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

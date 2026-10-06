# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v142 POR location correction bound

Focus: requested analysis found gain concentrated in conditional continuous POR density, not default gate or depth context. Test whether the successful width128 head is restricted by its fixed location correction bound. One candidate, locationbound2 vs measured v134 locationbound1; logweight bound remains1.

Explored/verified: prepared same319->128/GLU256->32 point head and same frozen v113 parent. Same82720 active parameters, random initialization order, seed, training sampling, original initial predictive function; only delta formula changes from s*tanh(u/s) to2s*tanh(u/(2s)). Both have unit derivative at zero; saturation differs away from zero. Newbranch bound only, original predictor/scales/default gate fully frozen. Reference s is fixed sourcecomponent width, not overall Student3/Cauchy mixture SD. Proper own POR likelihood/CDF/original grid/scorer unchanged. No v134 trained head loaded, no validation-derived bound fitting.

Gain: untrained/no score. Control v134 POR77.869090371913; compare restored best full144665 POR and actual combination fixedPERMv136/SWv104. Predefined default/nondefault diagnosis for interpretation only; full POR score chooses checkpoint. Correct for no conditional-only gain claim if defaults degrade enough to offset.

Not worth continuing: no location/logweight bound grid, defaultgate or depth-window sweep. Exis

R likelihood/CDF/original grid/scorer unchanged. No v134 trained head loaded, no validation-derived bound fitting.

Gain: untrained/no score. Control v134 POR77.869090371913; compare restored best full144665 POR and actual combination fixedPERMv136/SWv104. Predefined default/nondefault diagnosis for interpretation only; full POR score chooses checkpoint. Correct for no conditional-only gain claim if defaults degrade enough to offset.

Not worth continuing: no location/logweight bound grid, defaultgate or depth-window sweep. Existing width64/128 comparisons have capacity confound; this same-capacity bound comparison isolates one predefined formula. Larger range is a hypothesis, not guaranteed benefit.

Pending: explicitly launch after backbonev137 completed/handle missing; max2GPU alongside PORwide256v140, normal/80-18/min10/b64/64steps/3e-4/cos80*64. Atomic complete last/explicit resume/process-only keepawake. Final replay/frozen integrity and actual assembly; record wall/overlap/peak. Replaces PORhead training phase: source+POR113+POR142+PERM136+SW104=5 total. No new fullCV before strict originalfifthTOTAL>80.8+gain; OOF>82 unmet, old delivery unchanged.

Optional: after final useful POR mechanism evidence, predefine same head-factor comparison separately on PERM and SW within five total phases; no automatic extrapolation.

Requested POR gain analysis saved in results/deep_learning/por_joint_density_width_v140_20261001/GAIN_ANALYSIS_ZH.md. Both POR studies actually live, snapshot {"status": "running", "pid": 4352, "stage": "locbound2", "epoch": 19, "best": 77.59738834861389, "best_epoch": 17, "bad": 2, "peak_MiB": 922.2841796875}. Partial results only; first amplify measured POR mechanism then assess transfer to PERM/SW. Five phases, no extra folds/delivery changes.

V142 locbound2: restored POR=77.831487046825; delta vs v114=+0.283417551932, vs historical=+0.279550244940; actual fifth TOTAL=80.577837237427;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=34/52, seconds=379.708, peakMiB=922.284.

Closed locbound2: Locationbound2 loses to samecapacity bound1 by0.037603, close this bound extension. targetscore=77.8314870468251, best/stop=34/52, actualTOTAL=80.57783723742709. Exact144665/frozen replay; negative best/last cleanedMiB=19.861730575561523. Timing {"training_seconds": 379.7081502999936, "launch_to_report_wall_seconds": 385.394452, "peak_MiB": 922.2841796875, "overlap_other_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. No broad family/head failureclaim, no extra folds.



# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v126 POR output update scopes
Focus: continue exclusively POR until original fifth score>77.801936801885 (strict historical-best gain>0.25). Compare newly updating original default atom gate versus original mixture logweights alongside the retained successful location/side/router refinement. Fixed PERMv100 and SWv104.
Explored/verified: implemented gate/weight arms from exact v113 forward checkpoint, same predictive function as v114 likelihood-control initialization. Original locations/new heads/routes LR3e-5 unchanged; gate opens258 parameters, weight opens2064 parameters, all at3e-5. This is update-scope NOT capacity-matched. Proper hard expert0Student3/expert1Cauchy NLL/CDF original grid/actions, no new tail probabilities or utility losses. Frozen original backbones/reference widths; each arm keeps other newly tested factor frozen. No v114/v125 trained state loaded.
Gain: untrained, no evidence yet. Primary compare original-fifth POR vs retained v11477.54806949489308; also historical77.551936801885; actual three-channel TOTAL only after replay/assembly. Initial improvement from learning must not be confused with factor-specific gain.
Not worth continuing: closed width, PORutility, feature-thawing and negative SW scopes remain closed; no extra factors/LR sweep this batch. Do not infer whole factor class failure from one scope.
Pending: explicitly launch after v125 slots fre

-thawing and negative SW scopes remain closed; no extra factors/LR sweep this batch. Do not infer whole factor class failure from one scope.
Pending: explicitly launch after v125 slots free; max2GPU, normal, original9curves/206+allowedside/input-only bank/Q50/trainonly statistics/splits/seed/score,80-18/min10/b64/64steps. Atomic last/full best replay, then fixed other-target assembly. Five dependency phases originalsource/v113/v126/PERM/SW. No otherfolds below strict TOTAL>80.8+gain, OOF>82 unmet. Temporary keepawake, explicit early-stop/resume only. Record actual singles and pair wall/memory, not assumed speedup.
Optional: preserve measured factor gain and restructure next POR experiment within5 dependency phases; continue POR until >0.25 gain, no automatic queue or delivery edits.

V126 explicitly launched two concurrent POR jobs: gatePID56892 and weightPID67496; exact process command lines verified. Runtime snapshot {"gate": {"status": "running", "pid": 56892, "stage": "gate", "epoch": 8, "best": 77.54045432673898, "best_epoch": 0, "bad": 8, "peak_MiB": 920.85546875}, "weight": {"status": "running", "pid": 67496, "stage": "weight", "epoch": 3, "best": 77.54045432673898, "best_epoch": 0, "bad": 3, "peak_MiB": 920.8740234375}}; training snapshots only, no final gain. Other targets fixed; POR strict historical+0.25 milestone remains pending.

V126 gate: restored POR=77.540454326739; delta vs v114=-0.007615168154, vs historical=-0.011482475146; actual fifth TOTAL=80.487187761777;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=0/18, seconds=134.192, peakMiB=920.855.

V126 weight: restored POR=77.540454326739; delta vs v114=-0.007615168154, vs historical=-0.011482475146; actual fifth TOTAL=80.487187761777;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=0/18, seconds=132.467, peakMiB=920.874.

V126 finished both gate/weight best0/stop18 POR77.540454326739 vs v114 -0.007615168154, no new gain. All144665 replay/frozen exact, actualTOTAL80.487187761777. Closed this linear update scope, no LR/factor sweep; delete negative best/last 37.469MiB with SHA manifest. Pair timing {"concurrent": true, "pair_wall_seconds": 165.311885, "overlap_seconds": 110.503468, "single_reported_seconds": {"gate": 134.1915901999746, "weight": 132.46718979999423}, "peak_MiB": {"gate": 920.85546875, "weight": 920.8740234375}, "serial_equal_work_not_measured": true}; not a measured serial speedup. Resume two-route design: first new expressive joint POR head; second independently new random ConvGRU backbone.



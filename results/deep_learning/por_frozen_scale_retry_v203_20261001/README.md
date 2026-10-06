# POR frozen scale repair
Focus: preserve the verified three-stage learned density and test missing scale adaptation with small additive module; family T3/Cauchy and mixed density readout retained.
Explored: v200 failed beforeGPU due wrong parent path and remains immutable. V203 loads correct151 joint_rps SHA; all old parameters frozen, only8752 new parameters, LR3e-5, raw-scale bound0.1. No other repair loaded; initial predicted function identical151.
Gain: untrained, not proved; control POR77.92638076072386/TOTAL80.61247541799393. Optional sixth stage only if meaningful; default stays five. No fullOOF claim.
Do not continue: stage-ablation201 lost0.511034795872POR, so keep the three-stage base. No LR grid or scale expansion by default. No200 negative efficacy claim because it never trained.
Timing: report training including data/evaluation, overlap with202/actual full-work wall and memory. Counts source once; at most one added stage perhead.

V203 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds180.127673/peakMiB921.752930. No otherfolds.

V203 completed POR77.926380760724 best0/14, no learned gain, restored/frozen/replay/actualassembly done. Negative best/last removed with SHA manifest; no more scale-only scans on this POR setting. Keep151 effective stages and source. V205 remains running, early PERM89.408616231012 is not final.


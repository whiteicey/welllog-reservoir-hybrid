# SW tolerance-aware training targets

Focus: preserve positive probability-mass correction; change only conditional label representation to half hard-bin CE plus half normalized triangular physical5percent tolerance CE. Gate BCE unchanged. Uses true training labels only; no teacher or validation-derived targets.

Explored: prepared/untrained, no automatic launch. Same104 parent/12576parameters/zero init/LR1e-4/SAM0.1/utility0/fold5/Q50/80-14 as211. Physical bin representatives are existing physical density centers used by the unchanged decoder; if no bin gets nonzero utility, use hard-bin fallback. Decoder/bins/actions unchanged.

Gain: unproved; matched211 SW74.14762262262707. No trained211 weights; one added SW phase/six total, alternatives never stacked.

Not worth continuing: distribution feature213/source feature216 below repeat control; no more view expansion. Joint utility alpha1 was worse than alpha0; this candidate instead modifies supervised distribution targets, not utility-temperature/action search.

Timing: include data/warmup/eval, overlap/peak; actual restored prediction and full144665 three-channel assembly required. Default5 unchanged.

V219 final SW74.148179431612, gain+0.026426603454, actualTOTAL80.621724729203; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop7/21, trainingseconds304.111063/peakMiB908.494629. No otherfolds.

V211 actual optional6 TOTAL80.621724729203; actual optional7 (PERM207+SW219) TOTAL80.622804193366, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.

219 final SW74.148179431612, matched211 +0.000556808984. Actual optional7 TOTAL80.622804193366 (+0.000194883145 vs previous7), best7/stop21,304.111063s/908.494629MiB. Restored RAM-state replay exact/frozen exact; no diskreload claim. Positive kept, gain too small for smoothing grid/default extra phase.221 finished13 best3 SW74.122157243259 (vs104+0.000404415101); four-bias scope closed without broader conclusion.222 mass width128 and223 original65tap local filter are current two explicit GPU alternatives; no stacking/otherfolds.

Gain diagnosis,not trainingfit: PERM default66.037% scores97.278705->97.281652,continuous33.963%74.096912->74.113431; aggregate+0.007556. SW default69.669%98.715632->98.644703,continuous30.331%17.631889->17.881934; weighted gains-0.049416/+0.075842,net+0.026427. ContinuousSW zero-official-score66.389%,pred>=99.9 30.281%; latter is action range,not directgateclassification. Frozen-small-repair scope limits correction; representation/generalization hypothesis remains unproven, no noise/data-ceiling claim. Diagnostics saved CHANNEL_GAIN_STRATA.json and CONTINUOUS_ERROR_DIAGNOSTIC.json.

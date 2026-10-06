# SW attention distance bias

Focus: one small backbone mechanism; last global-attention head gets signed bounded2 distance bias, separation normalized by maximum token separation. Positive favors nearby, negative distant. All original104 weights frozen.

Explored: fresh version/untrained; zero bias, original continuous NLL/BCE+utility1/SAM0.1,LR1e-3,all blocks eval/noDropKey,originalfold5/Q50/80-14. Initial mathematical function unchanged but boolean-to-float mask may alter rounding; measure actual initial score separately. No218/211 trainedweights.

Gain: unproved,reference104 SW74.121752828158; only one added SW stage/six total,default5 unchanged.

Not worth continuing:218 firstQKV+trainDropKey best0/14 no gain; close that intervention, do not generalize to all attention.213/216 feature views closed.

Timing includes loading/warmup/eval and actual overlap,peak memory. Restore/replay and144665 three-channel assembly required; no exact trajectory claim for efficient attention backward.

221 uses explicit float32 QK/softmax/V in the modified last block after220 failed CUDA LSE backward alignment before epoch1. Original weights and zero bias kept; initial score separately measured for rounding.220 immutable failure, not negative evidence. No auto-resume.

V221 final SW74.122157243259, gain+0.000404415101, actualTOTAL80.612616963279; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop3/13, trainingseconds225.275674/peakMiB1211.433594. No otherfolds.

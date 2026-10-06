# v249 matched monotone transport

## Focus
Inspired by invertible spline transport (Neural Spline Flows, https://arxiv.org/abs/1906.04032); local variant is8segment piecewise AFFINE, not paper RQS/fullflow reproduction. Horizontal: push value on standardized logSW axis, compute oldCDF(inverseWarp(edge)). Vertical control: calibrate probability-axis oldCDF. Both4392params/zero-output uniformheights/mathidentity, sameparent/seed/optimizer/halfhard-halftolerance loss/SAM.

## Explored and validation
Prepared/untrained/held untilbothGPUslotsfree. Domain only existing train-fitted densityedges. Minheight.001, endpoints fixed, tails identity. CDF differences provide transformed binmass directly; no missing densityJacobian in bin likelihood. Freeze104/gate. InitialCDF float arithmetic may differ from historical component-wiseCDF helper: report initial score separately. No mixture-small-alpha attenuation. Final144665 ownSW/checkpoint/frozen/restoredRAM/actualassembly. Defaultgate frozen does not guarantee default action/score unchanged.

## Gain
Unproved. Horizontal-minus-matchedvertical plus absolutevs246control both required. OneSWaddition/sixalone. Max2, individual and pairwallclock recorded.

## Not worth continuing
247auxiliary currently tiny, no substantive amplification proved. Do not add more tiny-mixture/auxiliary coefficient scans if final negative. Vertical control tests probability calibration alternative, not a statement allCDF calibration fails. Research suggestions saved docs/deep_learning/research/continuous_readout_suggestions_20261001.md; no paper result implies local gain.

V249 vertical final SW74.127693242658, actualTOTAL80.614554563069; learned gain+0.005940552889, vs211-0.019929379969. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

V249 horizontal final SW74.124751573162, actualTOTAL80.613524978745; learned gain+0.002998883393, vs211-0.022871049465. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

V249 final: horizontal SW74.124751573162 (best2/stop16), vertical74.127693242658 (best1/stop15); horizontal-minus-vertical -0.002941669497. Both naturally patience14 stopped; neither beats retained SW24674.149145254062. Initial74.121752689769 separately reported; RAM-best restoration/frozen parameters/actual144665-row composition verified, not independent disk reload. Close this eight-segment affine transport configuration, no warp width/coefficient grid; no conclusion about all monotone transport. Positive minor gains kept. Concurrent pair wall 307.569s includes startup/data/model/initial evaluation, individual301.496/265.416s, excludes code preparation/CPU assembly; no serial speedup claim. Default5 and optional7 TOTAL80.624828025525 unchanged; no other folds or active GPU training remain. Next work: consider jointly adapting a small continuous-value representation and readout in one added phase, preserving retained controls; not yet implemented or launched.

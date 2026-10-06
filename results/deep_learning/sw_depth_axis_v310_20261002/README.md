# SW depth-axis joint representation v310

## Focus and hypothesis
Preserve fresh297well input-only FiLM, state branch, three MLP branches and shared density readout. After each original MLP add LN48, sigmoid Linear48 gate and zero-initialized DW residual with kernels3/7/15. Original104 is the only loaded trained prior. The new encoder, gates, conditioning and readout train together in one additional SW phase.

## Comparison and validation
Depth arm reads same-well neighboring valid depth positions with zero padding. Channel arm uses every identical tap at cyclic shifts of the48 latent channels at the same point. Both73953 trainable parameters; additional8688. Same tensors/init and initial valid function, different connectivity/function classes and padding behavior. No pooling, subsampling, depth wrap, or inactive center-mask control. Invalid residual inputs/outputs masked. Existing trainer checks real gradients, frozen signature, finite loss, population and restored whole-target best RAM replay. Independent disk reload reserved for qualification/delivery.

## Gains
Untrained; gain not established. Retained SW297well74.16003022301975, CONT17.502752767347182; historical stronger CONT25217.88486813991017. Current optional eight-stage total80.69562059394032 remains unchanged. Strict fifth total>80.8 plus useful gain required before remaining folds; full OOF>82 unmet. No defaultCE migration after309 CONT regression.

## Closure criteria and history
Not first depth-context experiment: historical TCN/GRU/latent and PERM273 gatedDW negatives differ in initialization/readout. Test this local adjacency setting only. If CONT fails the paired control or whole gain consumes CONT, close this setting rather than launch kernel/width/dilation grids. Single seed does not prove robust gains.

## Timing
At launch record two-process wall clock including data/initial evaluation/training/restored-best evaluation, excluding code preparation and CPU record. Record each job time/peak allocation separately; no serial speed claim from concurrent timings.

Research: [SW local depth gated](../cross_domain_research_20261001/SW_LOCAL_DEPTH_GATED_RESEARCH_20261002.md).

Explicit pair launched: depth PID85880, channel PID76952; max2 normal GPU; gains pending real results.

Running snapshot (not final gain): {"depth": {"status": "running", "pid": 85880, "stage": "depth", "epoch": 8, "best": 74.1349342495239, "best_epoch": 5, "bad": 3, "peak_MiB": 1390.0986328125}, "channel": {"status": "running", "pid": 76952, "stage": "channel", "epoch": 7, "best": 74.14086578545242, "best_epoch": 4, "bad": 3, "peak_MiB": 1714.6171875}}; both below retainedSW74.16003022301975; current optional8TOTAL80.69562059394032 unchanged. Training continues under own whole-SW best and patience14.

V310 depth: SW74.160199923047, actual TOTAL80.625931901205; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

V310 channel: SW74.155625583198, actual TOTAL80.624330882258; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

Final310 whole scores {'depth': 74.16019992304726, 'channel': 74.15562558319847}; depth-channel +0.004574339849; CONT paired -0.013838135467; strata {'depth': {'default': 98.84885176097337, 'continuous': 17.452652710549867}, 'channel': {'default': 98.83626121843403, 'continuous': 17.466490846016587}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 649.342s; no fullOOF or remaining folds.

310 closed: depth-channel whole+0.004574339849/CONT-0.013838135467; depth versus297 whole+.000169700028/CONT-.050100056797. Actual8 diagnosticTOTAL80.695679988950, delta+0.000059395010, strict808false, not adopted for continuous focus. Retained8TOTAL80.69562059394032 unchanged; evidence/checkpoints positive/control preserved, no generic kernel sweep. Pairwall649.342s, depth632.700s/channel644.247s; two-process timing not serial ranking.

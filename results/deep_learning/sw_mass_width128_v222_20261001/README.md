# SW probability-readout capacity

Focus: grow proven mass-readout nonlinear width32 to128,49536 parameters. Retain original104 frozen Gaussian prior,zero output initialization,bound0.5,gate frozen,own categoricalNLL+BCE/utility0/SAM0.1/LR1e-4.

Explored: prepared/untrained,originalfold5/Q50/80-14. Same211 recipe except width; control211SW74.14762262262707. Random hidden initialization shape differs,not same parameter ablation. No211 weights loaded; sibling from104.

Gain: unproved; reference211 gain+0.025870 vs104, continuous stratum+0.264388 with default-0.077974. Greater readout capacity may expand existing gain, no claim until best restored actual assembly.

Not worth continuing:213/216 extra views below repeated-hidden control,217 gate augmentation below211,218 firstQKV no gain; do not compound those modules. No broad width grid planned.

One added SW stage/six total; default5,optional withPERM207 seven; no otherfolds until TOTAL>80.8. Timings incl loading/eval/warmup,actual overlap,memory.

V222 final SW74.141314487722, gain+0.019561659564, actualTOTAL80.619321998841; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop7/13, trainingseconds245.395531/peakMiB933.126465. No otherfolds.

222 final SW74.141314487722, gainvs104+0.019561659564 but width128 below32 by0.006308134905; best7/stop13,245.395531s/933.126465MiB, close width expansion.223 finalSW74.121752828158 best0/14 no gain,271.268866s/1269.003906MiB; negative best/last deleted withSHA manifest after actual assembly, rest evidence retained. Current live224POR mass (initial77.926320949376 vs15177.926380760724, float renormalization -0.000059811348 not learned gain) and225SW lastFF residual. Max2GPU/default5/best optional7TOTAL80.622804193366, no otherfolds.

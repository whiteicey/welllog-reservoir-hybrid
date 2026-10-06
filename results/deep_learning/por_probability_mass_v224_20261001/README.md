# POR retained distribution-mass readout

Focus: transfer proven SW probability-mass correction to POR while retaining all positive151 weights and mixedT3/Cauchy prior. Freeze151, read its256hidden features;16672newparameters,zero-output,boundedlogmass0.5,gate fixed.

Explored: prepared/untrained. Own categoricalbinNLL+BCE+joint255edgeRPS1,LR1e-4/noSAM,originalfold5/Q50/256input/80-14. Nativemixedprior and original bins/actions/score retained. CategoricalNLL changes with representation; not pure architecture attribution. No moments assumed for Cauchy mixture.

Gain: unproved,controlPOR77.926380760724. Initialfloor/renormalization difference measured separately. OnePORextra stage,total6 baseline or8 with retainedPERM207/SW219; no other repair weights loaded. Default5 unchanged.

Not worth continuing: priorPOR scale/input-adapter additions failed; mass correction has evidence onSW instead. No existing positive stage discarded; POR stage ablation supported keeping113+151.

Record restored RAM replay/actual144665 assembly versus true disk reload,loading/eval/warmup,overlap and peakmemory. No otherfolds until TOTAL>80.8 plus gain.

V224 final POR77.926320949376, gain-0.000059811348, actualTOTAL80.612457474589; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds206.436611/peakMiB986.931152. No otherfolds.

224 finalPOR77.926320949376, best0/14, zero learned gain, original151 delta-0.000059811348 from floor/floatnormalization; actualassembly/replay/frozen exact.206.436611s/986.931152MiB. Negative best/last deleted withSHA manifest, original151 preserved.226 nativeNLL/BCE+RPS lowLR context-only fine-tune active, initial77.926380760724 exact parent score.225 requested earlyfinalization below211/219;227 input-conditioned residual prepared/noqueue. Best optional7TOTAL80.622804193366/default5 unchanged,nootherfolds.

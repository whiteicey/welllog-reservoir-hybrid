# SW amplitude-conditioned local backbone repair

Focus: transfer existing provenPOR amplitude representation into one small local backbone residual before global/detail paths. Freeze104;96hidden+63side ->32->96 zero output,8288parameters,0.1pointRMS bound. Old104 receives onlyoriginal206 inputs; same relative/well context.

Explored: prepared/untrained; amplitude/repeat predefined equalcapacity/initfunction candidates; side is existing compact100 amplitude.npz fitted on original64 training wells or first63 hidden channels. Effective function rank not matched. Ownoriginal continuousNLL/BCE+utility1/SAM0.1/LR3e-4,noDropKey,originalfold5/Q50/80-14. No prior repair weights.

Gain: unproved; parentSW74.121752828158. No fair amplitude-feature benefit until completed repeat control; first only amplitude explicitlaunch, repeatheld/no queue. OneSWextra stage/six total,sibling arms not stacked.

Not worth continuing:227 input-only wholewell conditioning best0/14; feature-specific local modulation differs.223 existing65filter fine-tune failed,210 finalhidden9tap featureadapter failed; this uses additional proveninput statistics earlier in backbone,not another kernel-size sweep.

Timing includes data/warmup/eval,actualoverlap/peak; exact restored RAM replay/frozen/144665 actualassembly required. No per-row oracle/testlabels/heldout fitting/decoder changes.

V229 amplitude final SW74.121752828158, actualTOTAL80.612475417994; learned gain+0.000000000000, vs211-0.025869794469. Six optionalphases, default5; no fair viewgainclaim until repeat/amplitude paircomplete.

227 finalSW74.121752828158,best0/14,no gain,242.930387s/1612.75MiB;228 finalPOR77.926380760724,best0/14,no gain,183.171263s/924.547363MiB.229 amplitude finalSW74.121752828158,best0/14,no gain,165.671455s/1397.282227MiB; repeat untrained/held as negativeamplitude does not warrant control cost, no featuregain claim. All actual144665 assemblies/replays/frozen exact; negative best/last deleted withSHA manifests, positive104/151/154 preserved.230PERM conditionalmass and231SW additiveBiGRU16 are current twoexplicitGPUjobs, default5/bestoptional7TOTAL80.622804193366 unchanged. No otherfolds.

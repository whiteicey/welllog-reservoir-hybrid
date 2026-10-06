# PERM v288 raw encoder width96

## Focus
Proven direct-supervised rawencoder architecture width48?96, residualMLP96?192?96?3/fuse288?96/direct256head.184000newtrainableparameters, alljoint3e-4. Samefixedalpha.3/5epochramp/directCE1/equalwellmixedNLL/RPS1 as283width48. Original154prior/gate fixed, no283/281weights/noextraPERMphase.

## Explored and validation
Prepared heldslots. Sameoriginalinput207/data/Q/seed/budget, seed20261271 preserved buttensor-shapes implydifferentnewcomponentinitialdraws; alpha0 samefinalprior. Capacityandrepresentationscale differs, notstrictfunction-preservinginitialization/no capacity-matchedclaim. Historical283positivecontrol89.493314539170 kept, absolute28189.498908539006. Not a repeat of272fullwarmexpert or273depth96: this scales the effective newlydirect-supervised pointencoder, not a weak nativehead or temporal alternative.

## Gains
Unproved. Predefine width96-minus-283 plusabsolute281delta andactualthreechannelassembly. OnePERMadditionalphase/sixalone/sevenwithSW285/eightifPOR287adopted, <=8. Originalfifthstrict808+gain beforeotherfolds/fullOOF82unmet/default5/deliveryunchanged.

## Not worth continuing
If96failsboth48controlandcurrentbest, close width96 ofthisrawdirectencoder config, no128/192/256grid. Priorfullbackbone negatives notgeneralizedtoallbackbones. Traininglossimprovement alone insufficient; reportofficialbest restore andactualpopulation.

## Timing
Atmost2GPU/normal, individualelapsed/peak andactualoverlapwall separately. Startup/data/initial/finalincluded, codeprep/CPUassemblyexcluded. LaunchafterSW285terminal only, noqueue.

V288 final PERM89.482470023653, gain+0.076905166734, actualTOTAL80.639392226351; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds201.223310/peakMiB797.685059. No otherfolds.

Final width96: PERM89.482470023653, best8/stop22; {"adopted": false, "versus_matched283": -0.01084451551712, "versus_best281": -0.01643851535349, "close_width_expansion": true}. Actual composition and RAM best replay/frozen proof, not independent disk reload. No remaining folds; fullOOF82 unmet.

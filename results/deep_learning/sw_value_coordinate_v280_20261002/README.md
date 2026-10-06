# SW v280 continuous coordinate energy

## Focus
New rawinput branch: original207 plus ninecurves17orderedneighbors and17validmask=377 input, MLP48/96/48,32coefficients. New valuecoordinate trunk1/32/32/32 at original256 finitephysicalgridcenters transformed with original training logSW median/scale; dot/sqrt32 produces categoricallogits. Both encoders/readout train3e-4 in onephase. Ordered coordinates vs seededshuffledcoordinates, sameparams/init/uniformcomponent. Neither oldbin/action order nor decoding changes. Fixed104 prior/atom protected by alpha0?.1 in5epochs; no trained275/278 repair weights.

## Explored and validation
Prepared/untrained. DirectconditionalCE1 halfhard/halfphysical5pcttarget independentofalpha, mixedconditionalCE/RPS1 as275. Aux CE already excludesdefaults; this tests target-value geometry and input-neighbor representation instead. Coefficients zero at initialization: firststep trunk/inputencoder gradients zero, second optimizerstep expected nonzero; same normaltrainer recordedgradient atstep1. Initial final prediction same prior. Relative/raworiginalfeatures retainunboundedinfo; addedneighborhood zero-pad invalid positions with explicitflags, no crosswell input. DeepONet-inspired factorization, not full operator reproduction or proven measurement physics. Capacity differs from275; pair isolates coordinate order only.

## Gains
Unproved. Predefine ordered-minus-shuffled and absolute SW25274.14922820781207. Restore best officialSW overall144665, actualcombinedTOTAL required. RAMreplay notdiskreload. OneSWaddedphase/sixalone/sevenwithPERM274, <=8. Originalfivefold82 unmet, strictfifth80.8+gain beforeotherfolds.

## Not worth continuing
Lowrank32 basis may smooth too strongly, shuffledinput might learn highfrequency functions, realgrid unequal spacing preserved. Positivepair does not prove neighborhood itself helps. If ordered failscontrol and retained252, close thisbasis/coordinate/ramp configuration; no basis-width/neighborhood/alpha grid. No oracleperrow selection/teacher/training-to-test labels.

## Timing
Individual and actual2concurrent wall separately. Include startup/data/initial/final, exclude codeprep/CPUassembly; peakmeasured notestimated. Prepared holds untilGPU slots free. Research source to be attached from independentagent.

## Observed failure correction
279 ordered initialeval invalid0/firstlossNaN because infinite terminaledges made midpointcoordinatesNaN. No validscore/trainedresult; shuffled279 neverlaunched. New280 source uses originalfinitephysical256centers, same initializationseed/data/loss/budget; preserves279source/failureevidence. Runtime finitecoordinate assertion covers actualobserved error, not a new preflight.
Research: ../cross_domain_research_20261001/VALUE_COORDINATE_OPERATOR_RESEARCH.md .

V280 both live verified; originalinitialSW74.121752689769 restored finitecoordinate fix,31248trainable. At epoch2 orderedbest74.125062885965/shuffled74.122079887753, peak1141.03125MiB each. Early snapshots not finalgain. Branch/valueencoder nonzero secondstep gradients recorded. No otherfolds.

V280 ordered final SW74.130726840713, actualTOTAL80.615616322388; learned gain+0.008974150944, vs211-0.016895781914. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V280 shuffled final SW74.128705295243, actualTOTAL80.614908781474; learned gain+0.006952605474, vs211-0.018917327384. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V280 final orderedSW74.130726840713/best5/stop19 vs shuffled74.128705295243/best4/stop18; fairdelta+0.002021545471, below252-0.018501367099. Actualall144665 assembly/RAM/frozenverified, no diskreload. Notadopted/nootherfolds. Perstrata diagnostics saved, notoracleselection. Timingwall255.320978s/individual249.536854,243.394834s/peak1141.03125MiB; codeprep/CPUassembly excluded. Close thisconfig, not wholeenergyfamily.

V280 stratifieddiagnostic: orderedcontinuous17.475277159184 vsoriginal10417.631889060480 andshuffled17.480330711490. Default98.796697 vsoriginal98.715632 creates overallgain. No continuousbenefit; noadoption. Finishedunpromisingpaircheckpoints deletedSHAmanifest, source/config/report/predictions/diagnostic retained, active281/282 andpositive104/252 preserved.

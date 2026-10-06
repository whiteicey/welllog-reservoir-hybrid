# v275 SW transfer of direct component representation learning

## Focus
Newraw207->48 stem/three48->96->48 residualMLPs/fuse144->48/sharedcategorical48->256 head (57568), alljointlytrain3e-4/nonzeroreadout. Original104prior/gate frozen. Same two initializednewfeaturebackbones+readouts differonlyfusion: probabilitymix alpha0->.1 vs priorrelativebounded3tilt gamma0->1, both5epochramp. Own independent hard/physical5pctsoft CE1 averagedover3branch+fused plus effective252mixedhardsoftCE/conditionalRPS1; aux continuous Q-normalized, no learnedgate. Same264PERMwellbalancedloss nottransferred. Newrepeat point-featureencoders, not263TCN/GRU.262/263 already hadcomponentCE.5, so this is not first directsupervision.

## Explored and validation
Prepared while272/274GPUactive. SWoriginal206+relative/1024window/Q50/64train16val/seed/80-14/b64/64steps/256bins800actions. Aux readoutlogits computedonlysupervisedmaskpositions tolimitmemory; sam0/dropout0 innewencoder, old104eval. Bothinitial use clamped normalizedprior withoutuniformfloor; discretizationdifference measured separately. Schedulemix_alpha buffer storesalpha, tiltgamma=alpha/.1, best/last includesstate; finalexactRAMreplay/actual144665combine notdiskreload. No numericalpreflight/independentreview/no teacher/no repairedweights.

## Gains
Unproved. Predefine mix-minus-tilt plusabsolute SWvs25274.14922820781207. Bothjointtrainfeaturebackboneandreadout, oneSWnewphase.6alone/7withPERM271, default5/deliveriesunchanged/strictTOTAL80.8+gain beforeotherfolds/fullOOF82unmet. Fusioncompare hasdifferentfunctionalgeometry/bounds, not matchedfunctionalcapacityclaim.

## Not worth continuing
262learnedmix alpha collapsed,263boundedprior-logtilt no learnedgain, both independentCE.5 existed; changeencoder/nonzeroreadout/aux1/fixedmix cannotisolatehistoricalcause. Thispairisolatesfusionundernewconditions, not allchanges. Frozenatomgate doesnotfreeze finaldefaultaction; newdensitymayharmlargedefaultpopulation. Ifnofairorabsolutegain, close thisconfiguration/noalpha-width-branchgrid. Do notcallCEdeclineofficialbenefit.

## Timing
Individual+actualstaggered/concurrentwall/peak, startup/data/initialeval/finalrestore included, preparation/CPUassemblyexcluded; no unmeasuredserialspeedup. max2GPU guard, armslaunchonlywhenslotsfree.

Research [SW direct component transfer](../../../docs/deep_learning/research/sw_direct_component_transfer_20261001.md).

V275 both actualGPUprocesses verified active; SWrawfeaturebackbone+directdistribution jointlytrain,57568 each, probabilitymix versuspriorrelativetilt. InitialSW74.121752689769 common vs104tinyfloor/discretizationdifference notgain. Staggeredlaunch/partialoverlap with274 recorded, max2/80-14/oneSWphase, nootherfolds. V272/273unpromisingweights deleted afteractualassembly/SHAmanifest, original154/positive271/current274/active275 untouched. CurrentactualsevenTOTAL80.649109567911 below808/fullOOF82unmet; default5/deliveryunchanged.

V275 mix final SW74.137236359285, actualTOTAL80.617894653889; learned gain+0.015483669516, vs211-0.010386263342. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V275 tilt final SW74.121752689769, actualTOTAL80.612475369558; learned gain+0.000000000000, vs211-0.025869932858. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V275 FINAL mixSW74.137236359285/best34/natural48; tilt74.121752689769/best0/natural14; pairedmixgain+0.015483669516, vsretained252-0.011991848527. Actual144665/RAMbest/frozenproof/no diskreload; retain usefulmix/control forfusionmechanism, notreplace252/noextraSWrepairchain. Individual454.294959/214.124284s/1242.015and1241.765MiB; staggeredpartlyconcurrentwall461.826s inclstartup/data/initialeval/finalrestore/excludesprep, notequal-startspeedcomparison. Diagnostic-only default/continuousstrata recorded, nottrainingweights ororacle. Currentactual7TOTAL80.649109567911/default5/deliveryunchanged/below808/fullOOF82unmet. V276numericPLE/scalarcompare body+readout bothjointtrain live/max2/80-14/onePERMphase.

# SW bounded probability-mass correction
Focus: test expressiveness beyond fixed8Gaussian components, preserving trained104 prior. Only small128->32->256 patch modifies original256bin logmasses by bounded0.5; gate/frozen model retained, all original256bins/800actions/scoring unchanged.
Explored: prepared/untrained. Zero output starts original probability function except1e-12 massfloor and float renormalization; report actual initial score and final-minusinitial separately. No gate/means/scales updates, no priorrepairweights. OwnSW categorical bin likelihood+BCE replaces continuous density NLL; original jointutility1/temp0.05 andSAM0.1 retained. Bound/logmasssoftmax maintains nonnegative normalized distribution. Train-only edges/sourcepreproc/Q50 unchanged; no targethistory/testlabels.
Gain: unproved; control104SW74.12175282815758/TOTAL80.61247541799393. Changed likelihood accompanies new mass representation, so not claim pure architecture ablation. Optional oneSWstage/six overall, alternatives do not stack; at most8total if three positive head repairs.
Not worth continuing:199Gaussian parameter repair best0/14 negative;208 specifically tests nonparametric bin correction rather than another location/scale sweep. No bin/action/decoder resolution tuning. If no gain, close this bounded setup.
Budget/timing:80/14/min10/b64/64steps normalGPU max2,LR1e-4. Record actual training wall including loading/validation, concurrency/total batchwall/peak; atomicstates/frozen checks/best restore/all144665 replay and actual3targetassembly required. No whole-recipe speed claim from patch timing.


V208 startup failed after GPU/data admission before epochs due manually mistyped parent SHA constant; actual104 diskSHA matches original199/plan07e0...f58670. Failure/source preserved, no efficacy score. V209 fresh source corrects only SHAconstant; independent explicitlaunch, no active trainedsource mutation.


V209 final SW74.142184487495, gain+0.020431659337, actualTOTAL80.619626498762; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds448.372218/peakMiB908.493652. No otherfolds.

V209 actual optional6 TOTAL80.619626498762; actual optional7 (PERM207+SW209) TOTAL80.620705962925, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.

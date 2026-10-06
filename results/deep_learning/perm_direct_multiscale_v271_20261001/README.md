# PERM v271 directly supervised raw-input multiscale distributions

## Focus
Original206+relative fresh trainable stem207->48, three MLP48->96->48 residual branches, fused144->48/sharedreadout48->256. Repeat scales1/1/1 vs multiscale1/4/16: same initialized structure/registeredcapacity, only masked average pooling differs. Direct auxCE from all3scales and fused distribution, nonzero categoricalreadout lets newrepresentation get first-stepgradient independently of alpha. Pure own labels, not teacher/sample retrieval; no claimed full TimeMixer reproduction. Prior154weights/atomgate frozen, final mixture fixed schedule alpha0 atinitial, epoch1..5 .02..1 then.1; not learned gate. No268/264 repairweights. Onejoint additional PERMstage.

## Explored and validation
Prepared while270twojobs active. Original train-only bins/Q/normal/originalfold5/80-14/b64/64steps/cosine80. Mixed conditional continuous-NLL halfrow/halfequalwell +jointRPS1, extra componentCE1/3 withsameQ. All branchaux average; no targetbalanced resampling. Pool valid within samewell windows/zero-padding excluded; pooledfeatures upsample withlinearinterp. Initial prior discretization clamped1e-12 may differfromnative154 and is measured separately. Alpha persistent buffer includedbest/last, set epoch for training before eachval; best0 or intermediate alpha is honest selected model state, not inference retuning. No extra pretraining phase. Original official256bin/800actionargmax kept. Full144665actualcompose afterrestore; RAMreplay notdiskreload.

## Gains
Unproved. Fair delta multiscale-minus-repeat; absolute PERM must compare26889.415840060079. Both share directcomponent/alpha schedule, their standalone benefit not isolatedbythispair. No autootherfolds untilstrictTOTAL80.8+meaningfulgain; fullOOF82 unmet.6alone/7withretainedSW252; default5/delivery unchanged.

## Not worth continuing
257point/local zeroqualityrepair gave local-negative0.00003358; not newnaiveTCN.268memory belowMLP/tiny absolutechange, no memorywidthsweep. SW263 newdirectspecialists had no learnedgain; this PERMtarget/directcomponentobjective+scaleinput is separately tested but still risky. Downsamplemaylose thinlayers, newdensity maylearn onlypriors, alpha mayharm currentknownfunction. If neither beatsretainedPERM, andscale nofairbenefit, close this exactpool/directbranch configuration; no alpha/scale/width grid. If CEimproves butofficialnot, do notcallrepresentation successful.

## Timing
Record individual andactualconcurrentwallclock/peaks; includestartup/data/initialeval/finalrestore, excludes preparation/CPUassembly. No assumed serial speedup. Research: ../../../docs/deep_learning/research/perm_direct_multiscale_distribution_20261001.md

Launched repeatPID89404/multiscalePID40228, normal initial89.4055666268939/same57568params. CUDAlinearupsampling backward emitted nondeterminismwarning withwarn_only; no trainingbitwisereproducibility claim, finalrestoredpredictioncheck stillrequired. No restart basedonwarning. Initialdiscretizationdifference notgain. This supersedes prepared/heldtext.

V271 final PERM89.439391359490, gain+0.033826502571, actualTOTAL80.624314693894; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds237.511100/peakMiB788.226562. No otherfolds.

V271 final PERM89.438759265115, gain+0.033194408196, actualTOTAL80.624093460863; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds239.721348/peakMiB787.983398. No otherfolds.

V271 FINAL repeatPERM89.439391359490/multiscale89.438759265115 bothbest8/natural22; multiscale-minus-repeat-0.000632094375, repeatvs268+0.023551299411. Original154 retained, directrepresentation+readout learning gain; extra poolscale notproven. Actual144665/RAM/frozenproof/no independentdiskreload. Sevenphase actualTOTAL80.633931076773, previous7delta+0.008242954794, stillbelow808/nootherfolds/fullOOF82unmet; default5/delivery unchanged. Bothpositive/controlweights retained. Individual237.511100/239.721348s/788.227and787.983MiB, concurrentwall247.236s inclstartup/data/initialeval/finalrestore/excludesprep. V272fullcopiedexpertencoder+directreadout live; v273fresh96temporalbackbone+directreadout launched besideit, max2/80-14/onePERMphase/nomutatedhistoricsource/norepairedweightschain.

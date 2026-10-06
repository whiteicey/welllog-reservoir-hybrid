# PERM v270 actual late backbone / native readout joint optimization

## Focus
Does representation adaptation improve over updating only the same retained154 native Gaussian readout? Prior repair-head gradients small but not proof backbone deficient. Start original154, retain its learned readout; no264/268 repair weights. Original SOURCE prior/atom gate stay frozen. Both arms copy the actual two expert dec0/combine terminal encoding modules, equal registered model capacity and initial predictive function. head_only copies remain frozen; late_joint jointly trains copies1e-5 and same154 readout3e-5. This is backbone feature adaptation for retained readout, not full original expert fine-tuning or architecture replacement.

## Explored and validation
Prepared, not trained; hold269twoGPU. Own original154 native Gaussian NLL/BCE plus jointRPS1, original decoder/Q/anchors/256window/splits. No uniform floor or discrete mass correction. Initial normal evaluation must equal154 PERM89.40556485691894. Late copies dropout eval with gradients enabled in joint arm; originals frozen. Atomic last includes RNG/optimizer/stopping. RestoredRAM final replay is not independent diskreload.

## Gains
Unproved. Predefined fair delta late_joint minus head_only; absolute comparison retained PERM26889.415840060079. One PERM phase, alone6/withSW2527; default5/deliveries unchanged. Strict fifthTOTAL>80.8 plus meaningful gain before same-config otherfolds, fullOOF>82 unmet.

## Not worth continuing
POR266 similar copied late joint update yielded zero gain, negative evidence for that target only. PERM267 changed its context head and mass loss but did not update actual source late encoding; this differs in native loss and encoding scope. If both degrade and best remains initial, close this update scope/LRs, no blind LR/width scan. No conclusion all backbones inadequate or unnecessary.

## Timing
Record both individual GPU times and concurrent process-creation-to-report wall time/peak; startup/data/initialeval/finalrestore included, preparation/CPUassembly excluded. max2GPU is upper limit, no serial speedup assertion.

Launched head_only PID87820 / late_joint PID71536 after269terminal verified. Both initial score exactly15489.40556485691894, trainable82720/117408, sourcecopies34688. This supersedes prepared/held text. No trained gain yet.

V270 live process commands verified bothGPUjobs, head_only/late_joint at epoch4 bestinitial89.405564856919/best0, no gain yet, eachpeak786.465MiB. Ownnative154head82720 versus head+actualcopiedlateblocks117408parameters; originalprior/gate frozen, no repairedheads loaded. Continue80-14/max2, onePERMphase/nootherfolds. Latestactualoptional7TOTAL80.625688121979 unchanged, strict808/fullOOF82 unachieved.

V270 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds145.626368/peakMiB786.465332. No otherfolds.

V270 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds147.725743/peakMiB786.465332. No otherfolds.

V270 FINAL bothPERM89.405564856919/best0/natural14/learnedgain0; actualTOTAL80.612475417994 olddefault5 replay, not improvement. Joint lateencodinggradient.05622446537/head.46505635977 nonzero, but nativehead/lateencoderadaptation didnotbeatinitial. Actual144665/frozenproof/RAMbest/no diskreload. Individual145.626368/147.725743s/786.465MiB, concurrentwall156.894s inclstartup/data/initialeval/finalrestore/excludesprep. Close thisscope at1e-5/3e-5, noLRgrid/broadbackbonefailureclaim. V271directcomponent rawrepresentation+readout launched repeat/multiscale same57568params, initial89.405566626894 slightly+.000001769975 fromdiscretization, notlearnedgain; onePERMstage/max2/80-14. Currentretainedoptional7TOTAL80.625688121979 unchanged/nootherfolds.

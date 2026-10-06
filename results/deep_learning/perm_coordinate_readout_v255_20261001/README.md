# v255 coordinate-conditioned PERM probability readout

## Focus
Replace free256bin projection in248 mechanism with small continuous-value-conditioned scorer. Query128->32GELU, train-fitted original log10 physical bin centers normalized[-1,1] into1->32GELU->32, interaction then shared32->1 zero projection. Bilinear query*value versus nonlinear tanh(query+value), matched5313params/init/loss. Separate query->alpha zero weight/init.001uniformbackground. Output logmass correction bounded.5. Numeric coordinates are output grid, not labels/test inputs.

## Explored and validation
Prepared/held until254terminal. All154weights/defaultgatefrozen. Retain248 normalized own continuous binNLL+jointRPS1/AdamW1e-4/80-14. Chunk512 checkpoint activation to fit max2GPU; no oversized backbone. Sameinitial zero output/.001background and own initialscore reported. Final RAMrestore/frozen/144665actualcomposition required; no diskreload claim.

## Gain
Unproved. Predefined nonlinear-minus-bilinear samecapacity and absolute retained248PERM89.413465595388. Parameters query4128+coordinate1120+projection32+alpha33=5313. Both readout mechanisms trained in ONE PERM phase, sixalone/sevenwith retainedSW252; no trained248/254repairweights.

## Not worth continuing
254 rawside yielded only tiny relative gain/belowretained, not sufficient for rawdepth/width scan. This changes numeric readout mechanism, not rawinput routing. If both fail, no coordinatefrequency/width grid. Distinct from prior sourceCDF transport193: this directly scores conditional binmass from frozen154 hidden plus output value coordinate.

## Timing
Record individual/pairactualprocesscreation-to-lastreport wallclock andpeakMiB; startup/model/data/initialevalincluded, codeprep/CPUassemblyexcluded, no serialspeedup claim.

Recorder-only entry fix: original copied mass-only recorder ignored CLI stage, produced no assembled score; training/source/config untouched. Use fresh record_perm_coordinate_readout_v255_r1.py --stage ARM. Actual composition pending r1 run; no invented verification.

255 actual launch bilinearPID39824/nonlinearPID69904, bothinitial89.404398334646/5313params. Epoch2bilinearbestinitial/nonlinear89.405634949173, early/unproved; both gradienttrain/noerrors/OOM, peaks790.252MiB. OnePERMaddition/max2, no otherfolds. Fresh record_perm_coordinate_readout_v255_r1.py handlesstage atfinal.

V255 final PERM89.405114632526, gain-0.000450224393, actualTOTAL80.612317839456; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/12, trainingseconds251.374102/peakMiB790.252441. No otherfolds.

V255 final PERM89.405634949173, gain+0.000070092254, actualTOTAL80.612499950283; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop2/12, trainingseconds250.074362/peakMiB790.252441. No otherfolds.

255 bounded12both: bilinear89.405114632526 best3 vsnonlinear89.405634949173 best2; nonlinearrelative+0.000520316647 only, absolute belowretained24889.413465595388. Smalllearned gains/controlweights retained, no coordinate width/frequency scan. RAMbest/frozen/full144665actualcomposition viarecorder_r1 complete; no independentdiskreload or convergence claim. Pairwall257.709s vsindividual251.374/250.074s/790.252MiB; includesstartup/data/model/initialeval, excludescodeprep/CPUassembly, no measuredserialspeedup.256 CE/q02 likelihood pair launched onlyafter255PIDs terminal,max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337/default5unchanged.

# v253 balanced optimization of state and continuous readout

## Focus
Retain246 multiplicative mass/background mechanism and219 halfhard-halftolerance ownCE. Separate new gate residual128->32->1, bounded0.5/zeroinit/seed20261110; masshead init seed20261010 identical246. All104weights frozen. Global versus balanced SAM matched16770params/init/loss/LR/80-14; only perturbation partition differs. Global radius0.1; balanced two independent group directions each radius0.1/sqrt2, same theoretical total radius0.1 when both nonzero. Group clipping2 botharms. No validation-derived trainingweights.

## Explored and validation
Prepared/held until252terminal. Gate4161+mass12609 trained JOINTLY in one phase, no trained246/238 weights. Original ninecurves/Q50/64trainingwells/fold5/256bins/800actions. Report first training gradient norms by group to examine hypothesis, not validationgradients. Final frozen104/RAMrestore/actual144665 composition, not diskreload claim.

## Gain
Unproved. Predefined balanced-minus-global and absolutevs246SW74.149145254062. Initialsame uniformbackground.001/zero gate residual; initialscore separately reported. OneSWaddition/sixalone/sevenwith retainedPERM, no stagechain.

## Not worth continuing
238 shared257head jointly gate/mass under globalSAM below independent mass219; it changed shared representation/output and seed geometry, not this matched separated-factor optimization. Current246diagnosis continuous+0.252758 versusdefault-0.070724 suggests cancellation, not proof gradient interference. If matched balanced no gain, no SAMrho or groupweight scan.252continues until terminal/min10.

## Timing
Individual and actual concurrent processcreation-to-lastreport wallclock; includes startup/model/data/initialeval, excludes prep/CPUassembly, no serialspeedup claim.

Prepared253 speed change: alpha=0 objective bypasses unused8192x800 softaction utility matmul; actual loss remains exact same symbolic likelihood. No active/completed source modified. Compare real epochwall/peak afterlaunch, no speedup claim yet.

253 real launch globalPID45612/balancedPID77268, matched initial74.121701580675/16770params. Epoch1global74.128904879153/balanced74.128795529416, early/unproved; both groups training/no errors/OOM, peak1119.827MiB. Epoch1about24.04s withconcurrentpair; bypassing unusedutility has no demonstrated speedup versus previousdifferentstudy, no attribution claim. Original104frozen, no otherfolds.

253 live verified epoch5: global74.131134840323/balanced74.132890964633, earlygap+0.001756124309/bothbelowretained25274.149228. First owntraining-only gradient norms mass0.005389/state0.008164; globalrho allocation0.05509/0.08346 versusbalanced0.07071each. Evidence of allocation change only, not proven gradient interference/gain across trajectory. DiagnosticJSON saved, no validationgradients/trainingweight fitting.254PERM matched raw/readout prepared/held while2GPUlive.

V253 global final SW74.131134840323, actualTOTAL80.615759122252; learned gain+0.009433259649, vs211-0.016487782304. Six optionalphases, default5; matched global versus balanced group SAM, one joint SW phase.

V253 balanced final SW74.132890964633, actualTOTAL80.616373765760; learned gain+0.011189383958, vs211-0.014731657994. Six optionalphases, default5; matched global versus balanced group SAM, one joint SW phase.

253 bounded matched12both/best5 finalglobal74.131134840323 versusbalanced74.132890964633, relative+0.001756124309, both belowretained25274.149228207812. Owntraining gradient partition changed, no material gain or convergence claim. RAMbest/frozen/full144665 actualcomposition done. Positive/control checkpoints kept; noSAMrho/groupweight grid. Concurrent pairwall318.879s vsindividual311.496/311.376s; startup/data/model/initialeval included,prep/CPUassemblyexcluded,nomeasuredserialspeedup.254PERM raw/repeat launched onlyafter253PIDs terminal,max2/onePERMaddition. Bestoptional7TOTAL80.624857059337/default5unchanged/nootherfolds.

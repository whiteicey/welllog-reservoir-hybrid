# v129 POR joint point versus positive depth-local control
Focus: matched-initialization ablation of positive v128 location head, replace33-tap depth reads with current-point reads. Identical stored kernel64x33 summed into effective1-tap kernel; all33 coefficients participate in gradient but effective kernel degrees of freedom collapse. Same parameter count and initialization, NOT identical effective function capacity or optimization trajectory; zero output guarantees same initial full prediction.
Explored/verified: implemented point_joint only, historical completed local v128 reused unchanged. Same frozenv113 parent/own mixed-familyNLL/width64/GLU/newbound1referencewidth/LR3e-4/windows/Q/80-18. No v128 trained head loaded. Same5 dependency phases, originalCDF/argmax/validation144665; fixedPERM/SW. No new source/coldbackbone in this POR route.
Gain: untrained. V128localPOR77.560504473312 (+0.008567671427 historical), actualTOTAL80.493202805749. Evaluate point-local ablation after finalrestoration, no early gain claims. Historical POR>77.8019368 target remains unmet.
Not worth continuing: weak v125learnedtails/v126linearoutputscopes closed, no bound/kernel/width grid.
Pending: explicit onePOR job alongside newbackbone(max2GPU). Restorebest/fullatomicstate, assembly and evidence/realwall/overlap/memory. StrictTOTAL>80.8+gain before remainingfolds; originalOOF>82 unmet, deliveries unchanged. Temporarykeepawake, no autoqueue/resume.
Optional: if depth reads provide real gain, retain for next isolated POR study within5 dependency phases.

Explicitly launched point_joint PID86428, exact livecommand verified; snapshot {"status": "running", "pid": 86428, "stage": "point_joint", "epoch": 23, "best": 77.55790267084286, "best_epoch": 16, "bad": 7, "peak_MiB": 922.1650390625}. Two sharedslots, no final gain yet, no otherfolds or delivery changes.

V129 point_joint: restored POR=77.561376174833; delta vs v114=+0.013306679940, vs historical=+0.009439372948; actual fifth TOTAL=80.493464316205;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=33/51, seconds=342.627, peakMiB=922.165.

V129 final pointPOR77.561376174833 best33/stop51, gain historical+0.009439372948, actualTOTAL80.493464316205;5 phases. Local-minus-point=-0.000871701521, no confirmed localcontext benefit; effectivekernelDOF differs. Retain both positivecontrols and pointbest/last, currentrecipepoint. POR+0.25 and TOTAL80.8 unmet.

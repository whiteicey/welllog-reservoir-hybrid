# v143 transfer conditional location/logweight head to SW

Focus: user requests expand measured POR gain then transfer to all outputs. PORwidth256 final77.912103 gives +0.043013 vswidth128; PERMwidth128 already +0.009542. Now test same nonlinear point conditional location/logweight mechanism on original SW Gaussian source.

Explored/verified: prepared319->128/GLU256->32 head,82720active parameters, frozen original dual source SW readouts/gate/scales/router. No v104 trainedSW network or other target head loaded. Two detached128expert hidden vectors +63 original-amplitude features. Location delta boundonefixedsource referencewidth, logweightdelta +/-1; zero newoutput initial sourcefunction. Own SW properGaussianNLL/BCE only, no extra utility, GaussianCDF and original SWactions/grid/scorer. This is same head mechanism transferred to another target, not Student3 family transfer, not a same-backbone/capacity comparison to randomTCNv104.

Gain: untrained/no score. Compare final SW to exact original-source initial score (direct evaluator) and retained v104Gaussian74.121752828158. Fixed verifiedPORv140/PERMv136, actual three-target assembly required; no fabricated TOTAL. Positive belowSOTA can support further isolated work but cannot promote remainingfolds.

Not worth continuing: SWStudent3 candidate early-stopped26/best25 at73.617843, belowGaussian0.503910, scoped weakresult; no df/loss/decoder sweep. PERM width256negative, do not automatically choose256 for allheads. Keep128 for this bounded transfer and properGaussianfamily.

Pending: explicit launch after PORv140 terminal/handle absent, max2GPU beside PORboundv142. Normal/80-18/min10/b64/64steps/lr3e-4/cos80*64, atomic complete last/explicit resume/process-only keepawake. All original9curves/206+63side/Q50/train64statistics/seed/splits/ownlabels/scoring unchanged. Final best144665 replay/frozenintegrity/actualassembly; record wall/overlap/peak. Replaces SWphase: source+POR113+POR140+PERM136+SW143=5total, no source+SW104+SW143 extra phase. StrictfifthTOTAL>80.8+gain before remainingfolds/fullOOF>82 notachieved, delivered package immutable.

Optional: retain useful SWhead mechanism, then fair factor-specific comparison at matched capacity. No automatic grid or inference mixing.

V143 final SW=73.622384767129, sourcegain=+0.000000000000, retainedGaussian delta=-0.499368061029; actualTOTAL=80.427243168594, five phases incl source;strict808=False, no otherfolds. best/stop=0/18, seconds=115.324, peakMiB=786.465; all144665replay/frozen exact.

Closed sw_conditional: SW head transfer best0, no gain versus frozen source; retain original SW104. targetscore=73.62238476712884, best/stop=0/18, actualTOTAL=80.42724316859398. Exact144665/frozen replay; negative best/last cleanedMiB=19.66194725036621. Timing {"training_seconds": 115.3244775000203, "launch_to_report_wall_seconds": 120.422902, "peak_MiB": 786.46533203125, "overlap_other_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. No broad family/head failureclaim, no extra folds.

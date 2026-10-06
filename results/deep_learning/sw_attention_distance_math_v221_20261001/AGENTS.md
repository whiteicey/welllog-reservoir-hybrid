Only last attention distance bias trains, all104 weights frozen. Originalfold5/Q50/score/80-14/max2GPU,one extra SW phase,no stacking/otherfolds. Maintain PROJECTS and evidence.

221 uses explicit float32 QK/softmax/V in the modified last block after220 failed CUDA LSE backward alignment before epoch1. Original weights and zero bias kept; initial score separately measured for rounding.220 immutable failure, not negative evidence. No auto-resume.

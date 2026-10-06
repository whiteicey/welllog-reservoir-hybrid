# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v139 SW Student-t3 density family

Focus: test a heavier-tailed proper conditional distribution in the successful compact local SW architecture. Change only Gaussian to Student-t3; keep 8 components, default gate, network/parameters, SAMrho0.1, utility1/temp0.05, LR3e-4, train sampling and Q50. Synchronize training NLL, differentiable utility interval mass, and inference CDF; retain original256 density bins/800 action grid/default/argmax.

Explored/verified: implemented Student3 scale=sd/sqrt(3), same finite component SD interpretation. Infinite edges replaced before arithmetic and assigned exact0/1 CDF values; own SW targetindex2 only. All SW weights random, no v104 checkpoint loaded. Same random-network construction and raw prior initialization, but initial predictive distributions differ; do not count initial family differences as trained benefit. Fixed current PORv134/PERMv136; actual final three-target assembly required.

Gain: untrained/no score. Control v104 Gaussian SW74.121752828158; current compact actual TOTAL80.589118234954. Same architecture and active parameter count, different family/loss and trajectories. Need restored-best all144665 SW prediction reproduction and actual combinedTOTAL, not partial training claim.

Not worth continuing: CDF Brier, old dilation and wider PERM route have closed negative evidence. Do not sweep df, utility/LR or decoder if this sin

amily differences as trained benefit. Fixed current PORv134/PERMv136; actual final three-target assembly required.

Gain: untrained/no score. Control v104 Gaussian SW74.121752828158; current compact actual TOTAL80.589118234954. Same architecture and active parameter count, different family/loss and trajectories. Need restored-best all144665 SW prediction reproduction and actual combinedTOTAL, not partial training claim.

Not worth continuing: CDF Brier, old dilation and wider PERM route have closed negative evidence. Do not sweep df, utility/LR or decoder if this single candidate fails. No label-derived inputs, teacher predictions or pseudo-labels; original9curves/206/relative/input-only wellbank/train-only64 stats/original splits and score.

Pending: explicit one SW job alongside backbonev137, max2GPU/normal/admission3GiB; 80/18/min10/b64/64steps/cos80*64, best restore and atomic last optimizer/scheduler/RNG. Process-only temporary keep-awake; record training seconds, wall, overlap and peak memory. This replaces SW phase, final source+POR113+POR134+PERM136+SW139=5 total phases. Strict fifthTOTAL>80.8+gain before remaining folds; original OOF>82 incomplete. No delivery overwrite or automatic queue.

Optional: retain distribution gain if measured, then bounded target-specific work within five total phases.

Live process and command verified, snapshot {"status": "running", "pid": 69512, "stage": "student3", "epoch": 1, "best": 71.19711508147167, "best_epoch": 0, "bad": 1, "peak_MiB": 1794.25732421875}. Both v137/v139 run in normal GPU, maximum2, no OOM. Supersedes prepared-only text. Partial scores, no final gain claim; current verified compact recipe remainsTOTAL80.589118234954/five phases. No remaining folds or delivery changes.

V139 student3: restored SW=73.617842845787; Student3 minus Gaussian SW delta=-0.503909982371; actual fifth TOTAL=80.412749741124;5 stages including source and2 POR phases; fixed PERM v136; strict808=False; no other folds. best/stop=25/26, seconds=612.391, peakMiB=1794.257.

V139 early stopped at26/best25: SW73.617842845787, retainedGaussian delta-0.5039099823705584; actualTOTAL80.41274974112382. Restore/replay/frozen exact; weak partial-bounded attempt, not proof of convergence or universal Student3 failure. Negative best/last cleanedMiB=18.204355239868164. {"training_seconds": 612.3913287000032, "launch_to_report_wall_seconds": 617.014697, "peak_MiB": 1794.25732421875, "overlap_other_backbone": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. POR priority replaces channel job, no resumption/folds.



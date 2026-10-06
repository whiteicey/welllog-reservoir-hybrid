# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v254 PERM local representation and retained distribution readout

## Focus
Keep248 normalized continuous categoricalNLL+jointRPS1/uniformbackground/.5bounded logmass mechanism. New side207 LayerNorm/depthwise9 residual concatenates with frozen154128hidden into335->32->257 readout. Raw arm: original206 train-fitted features+relativedepth; repeat control: frozen128hidden duplicated to207. Same21717params/init/seed/loss/80-14. No forbiddencurves/new externaldata/labels ininput.

## Explored and validation
Prepared/held while253GPU pairlive. Original154frozen/gatefixed, no trained248/239/245weights. Per-target native bins/actions andscore preserved. Side and readout trained jointly in ONE PERM addedphase. Initialzero output/.001background, initialscore reported separately. Final RAMrestore/frozensignature/144665actualcomposition, no diskreload claim.

## Gain
Unproved. Predefined raw-minus-repeat samecapacity mechanism and absolute versus retained248PERM89.413465595388. Sixalone/sevenwith retainedSW252, total<=8. Newparams335*32+32+32*257+257+414+2070=21717. AdamW1e-4/noSAM, originalQ50/64train16validation wells, ownPERMlabels.

## Not worth continuing
239 local hidden filtering below230 and245 rawsideSW below repeated-hidden control; this transfers rawbranch to PERM with separately-normalized effective readout mechanism, not a claim raw features alreadyhelp. If both below retain

dphase. Initialzero output/.001background, initialscore reported separately. Final RAMrestore/frozensignature/144665actualcomposition, no diskreload claim.

## Gain
Unproved. Predefined raw-minus-repeat samecapacity mechanism and absolute versus retained248PERM89.413465595388. Sixalone/sevenwith retainedSW252, total<=8. Newparams335*32+32+32*257+257+414+2070=21717. AdamW1e-4/noSAM, originalQ50/64train16validation wells, ownPERMlabels.

## Not worth continuing
239 local hidden filtering below230 and245 rawsideSW below repeated-hidden control; this transfers rawbranch to PERM with separately-normalized effective readout mechanism, not a claim raw features alreadyhelp. If both below retained and raw belowrepeat, close this bounded input branch, no depth/width scan.

## Timing
Record individual and pair actual processcreation-to-finalreportwallclock, peaks/startup/model/data/initialeval included, codeprep/CPUassemblyexcluded. No serialspeedup claim.

254 actual launch rawPID2980/repeatPID50424, sameinitialPERM89.404398334646/21717params, max2. Original154frozen/no repaired248weights; onejointPERMstage. Initialeval completed/no errors; no gain yet.

Recorder-only entry fix: original copied mass-only recorder ignored CLI stage, produced no assembled score; training/source/config untouched. Use fresh record_perm_raw_readout_v254_r1.py --stage ARM. Actual composition pending r1 run; no invented verification.

V254 final PERM89.410293214370, gain+0.004728357451, actualTOTAL80.614130343102; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop4/17, trainingseconds187.801940/peakMiB790.049316. No otherfolds.

V254 final PERM89.409306142997, gain+0.003741286078, actualTOTAL80.613784868121; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/17, trainingseconds188.345957/peakMiB789.947266. No otherfolds.

254 bounded matched17both finalraw89.410293214370 best4 versusrepeat89.409306142997 best5, relative+0.000987071374 tiny, both belowretained24889.413465595388. Source/frozen/RAMbest/full144665 actualcomposition done viafresh recorder_r1, old mass-only recorder ignoredstage/no score; training source unchanged. Positive/controlweights retained; no rawdepth/width scan. Pairwall196.773s, individuals187.802/188.346s, peaks790.049/789.947MiB, startup/model/data/initialevalincluded/codeprepCPUassemblyexcluded/no serialspeedup claim.255coordinate querypair launched onlyafter254PIDs terminal, max2/onePERMaddition. Bestoptional7TOTAL80.624857059337unchanged/default5preserved/nootherfolds.



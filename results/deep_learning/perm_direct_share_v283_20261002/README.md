# PERM v283 fixed30 share training

## Focus
281ramp-to50 selectsbestepoch3 actualalpha30 andbeats274; later50share worsens. Testpredefinedfinal30with5epochramp, sameoriginal154prior/newrawencoder/readout/init/loss/LRs/budget. No281/274trainedweights. Newencoder/readoutjointlytrain, onePERMaddedphase.

## Explored and validation
Prepared fromobservedtrajectory, notposthoc inferencealphascan. Historicalcontrols27489.482758477027/28189.498908539006. 281selectedalpha30at3, versus new30reachedat5; schedules differ, notsameearlyphase rerun. Bestbufferalpha preserved, actualstoredalpha/epoch mustreport. Originalfold5/Q/9curves206/80-14/max2/nootherfolds.

## Gains
Unproved. Predefineabsoluteimprovement vs281 plusactualPOR151/SW252assembly. OnePERMaddedphase/sixalone/sevenwithSW252, <=8. Default5/deliveryunchanged, strict808+gain beforeotherfolds, fullOOF82unmet.

## Not worth continuing
If final30doesnotbeatselected281, close localshare amplitude exploration here; no.35/.4/... inferencegrid. Do notcall281selected30 a50share effect. Candidate maygivefairmechanismgainbelowSOTA butneeds actualrestoredcomparison.

## Timing
Atmost2GPU, individualstartup/data/initial/final included and actualoverlapwall recorded; codeprep/CPUassemblyexcluded. No serialspeedupclaim.

V283 final PERM89.493314539170, gain+0.087749682251, actualTOTAL80.643187806782; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds307.513264/peakMiB788.226562. No otherfolds.

V283 finalPERM89.493314539170/best8/stop22, vs281-0.005593999836, vs274+0.010556062143. Notadopted, usefulfixed30 positivecontrol kept, close localsharegrid. Bestcombo28180.654762 preserved/nootherfolds.

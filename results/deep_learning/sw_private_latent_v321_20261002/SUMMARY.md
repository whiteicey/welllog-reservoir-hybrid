# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V321 private cross-target input latents for SW

Focus: strengthen representation using POR151/PERM154 learned private context features, not shared SOURCE hidden or output predictions. NativeSW104 prior fixed; fresh297well raw encoder/state/FiLM/readout jointly train with21600 new hidden fusion,86865 totalactive. Only original151/154/104 weights load, no trained297/otherrepair. All originals frozen/eval; no teacher/KL/distill/label retrieval/prediction voting.

Control: both compute original151/154 donors for equalforward cost. Own fusion receives SW104.output-before128 hidden repeated3 to384; cross receives POR151.expand-before256 +PERM154.expand-before128. Same LN384/Linear384?48/GELU/zeroLinear48?48, added .25tanh after newstemFiLM. Active count/init equal; effectivefeature rank/information and native context differ, do not claim equalfunctionclass or causalproof of superiority of learnedlabels. Source priors themselves were selected on originalfifth, not independent OOF.

Alignment/isolation: SW1024 prior; donor originalfixed256 central slice (lo=L/2-128), sameoriginalrow indices/relativeanchor/absolute/valid. Both donorlatents covercentral256 including128 scoring/training core, paddedzero elsewhere only in freshpointwisebody. Do not send1024 to originalSOURCE (215 failed that contract). POR/PERM receive existing206+amplitude63 cache, SHA4935b88a194e825f8794ff3e0125ecad8d5

 botharms, pairwall/stagger distinctfrom individual; measuredGPUpeak, no tinyparametercheapness claim.

Done: sources/plan/README prepared, localsyntaxcompiled. Next: actual onejoblaunch then measuredmemory/free-slot admission forcontrolpair; no numericalGPUpreflight. Unfinished: real gain,808qualification/fullOOF82. Optionalpostgoal: independent replication if warranted.

INITIAL START FAILURE: original321 cross PID86524 failed before initialscore/training due privatecapture method name colliding with inherited fused forwardhook signature. No epoch/checkpoint/score. Originalfailed source/config/log preserved. New321r1 renamesprivate hook factory only, same mechanism/seeds/config, new outputfold5_r1; not a trained repair/additionalphase.

LIVE321r1 confirmed crossPID89936/ownPID75612, both initialSW74.12175268976877. Active86865 matched params, newbody/state/FiLM/readout/fusion jointly train, threeoriginalsourceencodersfrozen. Cross firstmeasuredpeak1560.181MiB; own startedinitialverified. {"own": {"status": "running", "pid": 75612, "stage": "own", "epoch": 1, "best": 74.12175268976877, "best_epoch": 0, "bad": 1, "peak_MiB": 1560.1806640625}, "cross": {"status": "running", "pid": 89936, "stage": "cross", "epoch": 4, "best": 74.12798751189538, "best_epoch": 3, "bad": 1, "peak_MiB": 1560.1806640625}}; live snapshot only, no finalgain/adoption/fullOOF/otherfolds. Atlas updated1174experiments/233deadends including320closure and321startupfailure+r1.

V321 own: SW74.166030180492, actualTOTAL80.627972491311; fresh297 raw encoder/state/condition/readout +21600 latentfusion joint; original151/154/104 private hidden sources, no donor predictions, oneaddedSWphase. No fullOOF/otherfolds.

V321 cross: SW74.166591356036, actualTOTAL80.628168902751; fresh297 raw encoder/state/condition/readout +21600 latentfusion joint; original151/154/104 private hidden sources, no donor predictions, oneaddedSWphase. No fullOOF/otherfolds.

FINAL321: both best13; early22own/25cross. OwnSW74.166030180492/CONT17.911355427690, gainvs297 whole+.005999957472/CONT+.408602660343, CONTvs252+.026487287780. CrossSW74.166591356036, cross?own whole+.000561175544/CONT?.280224707870; close cross migration, retain own. Actual8 {"POR": 78.03055279414464, "PERM": 89.51580440487884, "SW": 74.16603018049169, "TOTAL": 80.69780794312307}; strict808false/nootherfolds/fullOOFunmet. Newrecipe inthisstudy, old312/default5/deliveriesunchanged. Individual690.157/741.981s peak1560.181MiB, pair746.514s, campaign793.156s inclfailedstartup/fix; no isolatedspeedranking or independentdiskreload. No width/fusiongrid.



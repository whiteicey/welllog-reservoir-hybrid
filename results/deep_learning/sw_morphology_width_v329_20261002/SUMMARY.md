# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW positive morphology full fresh-body width96 v329

## Focus / controls
One capacity expansion of retained326first morphology+ownlatentearly joint encoder/readout: freshhidden48->96, MLPinner96->192, stem207->96, 3branches/fuse288->96, density96->256, gate96->1, own128repeat3->LN384->96->96. Input-onlybankpoint32 andstats96 remain, onlyFiLMcode output192. Firstbranchsamek7max/min, LN96/SE96x7x2/projection192->96zero, same .25bound. Original104845913prior frozen/only checkpoint, no326/327trainedrepairweights. Allfreshparametersjoint inoneSWphase/6standalone/max8.

## Explored / validation
Research [width audit](../cross_domain_research_20261001/SW_POSITIVE_MORPHOLOGY_WIDTH96_RESEARCH_20261002.md). HistoricalSW222frozenmasspatch hiddenwidthnegative/SW57nativeFFpatchnegative, PERM288rawwidthnegative lowerexpectation but notsameSW326wholefresh intersection. Notclaimfirstwidthsweep. Actualarithmetic count262017=stem19968+3MLPs111456+fuse27744+readout24832+gate97+condition10816+ownfusion47040+morph20064; about2.839xnewparameters despite2xwidth. Total1107930withprior, noOOMestimateclaim.

Replacedmodules require hooks restoredexactly: stem wellFiLM then ownlatentearly, branch0morph once, fuse statecapture; source104ownnormcapture retained. Allinitialzero outputmodules retainoriginalpredictor atalpha0; freshnewencoderfunction differsfrom48, notfunction-preservinggrowth. No numerical

e104ownnormcapture retained. Allinitialzero outputmodules retainoriginalpredictor atalpha0; freshnewencoderfunction differsfrom48, notfunction-preservinggrowth. No numericalpreflight/review; ordinaryinitialscore/count/frozen/bestRAMreplay inactualtraining only; syntaxcheck.

## Gains / comparison limits
Untrained/no gain. Historical32648 natural27/best13 SW74.169543032075/CONT17.963442088248. Newnominal80/pat14/min10/b64x64/cos80, savebestthrough27. Onlypredefined27ceiling compares equalepoch; laterfullbest absolutecandidate notsameworkgain; naturalstopbefore27 actualworksmallerdisclosed. Capacityandinitfunctions differ; cannotattribute to morphologycausalbenefit or scalinglaw. Q50/seed20261010/original9/206+relative/64train16val/score/alpha.25ramp5/loss/decoder unchanged.

## Close / cost / next
Onlyone96candidate, notwidth/LR grid. RequireusefulwholeandCONTvs326first, otherwisekeep48 andclosewholebodywidthdirection. No phasechain/oldweights/modelvoting/teachers/pseudo/testlabels/otherfolds orautomaticqueue. Peak/individualwall/paired27workcost tobe measured, includesprep/initial/train/bestrestore, excludesimplementation/finalization; noisolatedspeedclaim. 12GB/max2actualprocessconstraint; no launchuntilfreeGPUslot. Current8TOTAL80.699037441177/strict808false/fullOOF82unmet.

V329 wide explicitlylaunched PID75584 afterfreshCIM showedonlyPOR328PID43860; initialSW74.121752689769/actual262017newparameters verifiedduringordinarytraining, no extraGPUpreflight. Bothareonly2GPUtasks, max80/pat14, gainunproved; bestthrough27historical48comparison defined, no326weights/otherfolds.

V329 wide: SW74.131552954773, actualTOTAL80.615905462309; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure and recorder description correction
Natural stop15, restored best1 SW74.131552954773, continuous17.54497656542636. Relative retained326 width48: whole -0.037990077302, continuous -0.418465522822. No adoption. Actual NEW262017 parameters: stem/three96-wideMLPs/fuse/readout/state/FiLM/own128-to96 early latent fusion/96-wide first-branch morphology. The automatic recorder ancestor text calling this a 48-wide late fusion is incorrect; CONFIG/source and this description are authoritative. Only original104 trained weights loaded. Stopped before predefined27: smaller actual work, not an equal-work scaling gain; no width/LR grid. Individual394.467424s, peak2336.997MiB, concurrent POR328. Best144665 RAM replay/frozen exact, not independent disk reload or completeOOF.



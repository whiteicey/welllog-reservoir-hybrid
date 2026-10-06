# SW bounded variance-DSU v334

## Focus / controls
Onlyoriginal104trainedprior +fresh326positive architecture92289 jointnewencoder/readout parameters. Trainingonly after stemwellFiLM+own104latentearlyfusion before branches: maskedwindowmean/std detach, uncertainty=crossbatchstd of windowstd; independent Bx1x48 noise, perwindowp=.5, delta=.1tanh(noise*uncertainty/std_floor1e-3). h+(h-mean)*delta preserves mean, standarddeviation variation atmost10%; invalidrowszero. No labels/validation statistics, originalprior untouched, no extra trainableparam or phase. Identity computes same statistics/draws and returns originalh exactly; evaluation hook inactive both. Allfreshbody/readout/state/FiLM/latent/morphjoint, alpha.25 unchanged/own wholeSWbest.

## Explored / validation
[Primarypaper/officialsource and duplicate audit](../cross_domain_research_20261001/SW_VARIANCE_DSU_RESEARCH_20261002.md). MixStylev51/v189/v267 negatives limit expectations; this is mean-preserving variance-only bounded DSU migration, not fullDSU reproduction/newclass. Windowstd variation may reflect true geology/label information or coverage, not verifieddomainuncertainty. Perturbation may remove useful signal; no crosswelllabelmix/teacher/pseudo/externalweights. Original9curves206relative/fixed64train16val/seed20261010/Q50/decoder/80pat14/b64x64/cos80 unchanged. No preflight/review; ordinaryinitial/frozen/RAMbest and own-target score only.

## Gains
Prepared/untrained/no gain. Historical326 SW74.1695430320749/CONT17.96344208824835; identity-vs-noise pair predefined. Predeclaredbestthrough27 duringnormaltraining, historical326natural27; later best not equalwork gain and earlystopactualworkdisclosed. No adoption unless whole and continuous positive against retained326, positivepairalone mechanism evidence.

## Closure / costs
Only two namedarms, no probability/strength/meanvariance/LRscan or automatic queue/otherfolds. OneSWphase/6standalone/max8 and max2globalGPU. Preserve deliveries/default5/retained8TOTAL80.699037441/fullOOF82unmet. Record actualindividual/pair wall/stagger/concurrency/peaks; prep initial train restore included/implementation CPUfinalization excluded.

334 launched identityPID47368/noisePID36948 after333 terminal, actualmax2GPU globaladmissions. Normalenvironment80pat14/ownwholetargetbest/oneSWphase/jointfreshbodyreadout preserved; not gainyet.

V334 noise: SW74.163475724720, actualTOTAL80.627078431791; original104only/fresh326earlylatent+morph joint encoder/readout training-only mean-preserving boundedvarianceDSU,92289 active, oneaddedSWphase. No fullOOF/otherfolds.

V334 identity: SW74.169543032075, actualTOTAL80.629201989365; original104only/fresh326earlylatent+morph joint encoder/readout training-only mean-preserving boundedvarianceDSU,92289 active, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure
Both explicit evidence-based earlystop25/best13 (not natural27). Identitywhole74.1695430320749/default98.63988120716091/CONT17.96344208824835, exactly matches historical326. Noisewhole74.16347572471956/default98.63999486415061/CONT17.943177678122417. Pairedwhole -0.006067307355/CONT -0.020264410126, no adoption. Stoprequests recorded at24 because bothbest13/11nonimproving and noise loseswholeCONT; source/config unchanged and samePIDsrestore/RAM144665exact/frozenexact, notindependentdiskreload. Bothactual25 fair paired scope, predeclared27 snapshots not reached/no27gain claim. Close varianceDSU without strength/probability/meanvariance/LRgrid. Pairwall589.518556s/stagger1.102s, individuals585.006699/583.524759s, peaks1779.212/1779.998MiB, prepinitialtrainrestoreincluded/implementationCPUfinalizationexcluded. Identitypreserved retained positive control/reproducibility, not anothergain. Retain8TOTAL80.699037441/default5/fullOOF82unmet/nootherfolds.

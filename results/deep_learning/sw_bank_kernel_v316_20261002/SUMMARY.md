# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW bank conditional feature kernels v316

## Focus
Change the three fresh297well MLP48-to96 feature matrices using a same-well input-bank routing vector. Preserve original effective state/FiLM/density/direct-CE mechanisms and original104 prior; all fresh modules jointly train in one addedSWphase. CondConv-inspired matrix kernel migration, not full ConvNet reproduction or pretrained weights.

## Predefined comparison and verification
Each branch adds sum_e pi_e U_e V_e with four rank8 bases (U96x8,V8x48). One shared router96-to16-to4 reads mean/std/max of the existing BankCondition.point32 bank features. Actual-bank versus fixed train-input reference only changes new router source; original FiLM and prior remain actual-bank in BOTH arms. Same80709 active parameters,15444extra; reference effective function class has constant-input redundancy, not strict expressivity matching. No four token/member paths or probability ensemble: mix one Bx96x48 feature matrix before applying one MLP. All original widths, loss, alpha.25/statebound4/readout/mask/splits preserved.
Uzero, distinct nonzeroV, routerlastzero gives uniform pi and identical initial function. First step U updates while V/router kernel gradients may be zero; routerlast can learn second step if expert responses differentiate, first layer usually after third. Shared bankpoint also learns via FiLM. Actual update evidence is re

 gap. No315repairweights/allocation module used.

V316 live pair confirmed: actualPID16564 epoch22/best74.15415795130484@17; referencePID37364 epoch5/best74.14116013736248@4. Both original104init74.12175268976877/80709params verified, no stderr errors, max2realGPUjobs. Reference launched explicitly after315scalarterminal,227.14s gap. Snapshotnotfinal, no matchedgain/adoption claim. CPU finalizer now reconstructs exact seeded U/V/router initialization and measures selectedbest deltas so initially nonzeroV/router hiddenweights are not mistaken for updates. Fullpairfinal/strata/composition pending; retainedoptional8TOTAL80.69570795800789/default5/delivery unchanged, strict808/fullOOF82 unmet.

V316 actual: SW74.154157951305, actualTOTAL80.623817211095; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V316 actualfinal74.15415795130484/best17/stop31,CONT17.47294302201801, bothbelow297. Referencebest74.15413935754133@17; earlyboundary request becausecandidateclosed andtinywholepairedbenefit, notgeneralCondConvfailure. V317relu34568explicitlaunched, original9percurveL/LR3600extra/68865params; linearstillprepared, max2.

V316 reference: SW74.154139357541, actualTOTAL80.623810703278; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final316 whole scores {'actual': 74.15415795130484, 'reference': 74.15413935754133}; actual-reference +0.000018593764; CONT paired -0.001590054584; strata {'actual': {'default': 98.83134555555758, 'continuous': 17.47294302201801}, 'reference': {'default': 98.83062660776832, 'continuous': 17.47453307660211}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 656.926s; no fullOOF or remaining folds.

V316 finalclosed: actual74.15415795130484/reference74.15413935754133, bothbest17/stop31 NATURALpatience14; referenceearlyrequesthadnoeffect, notearlyterminated. Actual-referencewhole+.000018593764/CONT-.001590054584, bothwhole/CONTbelow297/252, notadopted/no kernelgrid. AllU/V/routerselectedbestdelta normsfromexactseededinitialization nonzero, notdeadmodule. Pairwall656.925814s, individual497.601626/424.509887s/1284MiB,stagger227.14s/variableoverlap; noisolatedspeedclaim. Sourcefrozen/RAM144665replayexact, notindependentdiskreconstruction. V317 relu34568/linear52776 explicitlylaunched,90.282s gap, original104only/newbody/readoutjoint, no316weights. Retainedoptional8TOTAL80.695707958008 unchanged, nootherfolds.



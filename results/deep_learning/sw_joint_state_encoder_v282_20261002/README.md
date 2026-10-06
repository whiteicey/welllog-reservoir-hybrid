# SW v282 state and density feature co-learning

## Focus
New57568rawpointencoder+direct256density as275mix, plus49param zero-init bounded2 defaultgate residual fromsame48fusedfeatures. Allnewencoder/density/state train jointly3e-4, original104 weights frozen. Onephase from104, no275/269trained repairs. Samealpha.1 ramp5/directcontinuousCE1/mixedCE+conditionalRPS1. Predictivedefaultgate maychange throughresidual; priorweightsfrozen does not meanstatepredictions frozen.

## Explored and validation
Prepared finite twoarms: ordinary versus balanced classBCE as269 trainQprior correction; onlyBCEweighting/shift differs, priorfittraininglabels only. Inference removeslosslogitshift. DensityauxCE alreadycontinuous-only normalized and unchanged. Historical269state mass repair had +.01014 balancebenefit but below252; here newrawencoder learns bothstate/density, not frozen128features. Bothsameinitialgate/mass, sameparams/init/data/seed/budget. OfficialoverallSW selectscheckpoint; default/continuous strata diagnostics only no oracle.

## Gains
Unproved. Predefine balanced-minus-ordinary plusabsolute retained25274.14922820781207; compare275density-only74.137236 ascontext (not identicalparameter control). OneSWadditionalstage/6alone/7withPERM274, <=8. Strictfifth808+gain beforeotherfolds, fullOOF82unmet/default5/deliveryunchanged.

## Not worth continuing
If jointstate cannot beat retainedSW norbalancedcontrol, close thisbound2/raw48/commonfeature setup; no gatebound/BCEweightgrid. Sharedstate/continuous losses can conflict; balancedBCE improvesclassification surrogate only and mayharmofficial continuousdecision. Neverequate finaldefaultaction togateaccuracy.

## Timing
Individualactualelapsed/peak plusactualstaggeredpairwall; startup/data/initial/finalincluded, codeprep/CPUassemblyexcluded. At most2GPU. Explicitlaunch after280slotsfree; mayoverlapPERM281, noqueue.

Live verified ordinary: epoch8, best74.12175268976877; not finalgain. Atomicbest selects currentmixalpha too. Nootherfolds.

V282ordinary boundedstop requested after15+ epochs; best74.122510 belowretained252/275. Compare balanced with same actualcompleted epoch budget once ordinaryterminal; no convergenceclaim or unequalbudget finalfairgain.

V282 ordinary final SW74.122510119287, actualTOTAL80.612740469889; learned gain+0.000757429518, vs211-0.025112503340. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V282 balanced final SW74.143119674512, actualTOTAL80.619953814218; learned gain+0.021366984743, vs211-0.004502948115. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V282 final ordinarySW74.122510119287/best14/bounded18 vsbalanced74.143119674512/best17/bounded24. Matched18prefix samebest17 state/pred exactlymatches finalmodel, fairdelta+0.020609555225; actualoverrun24 not concealed. Below252-0.006108533300, noadoption/nootherfolds. Positivebalance/control/matchedproof kept; actual144665assembly/RAM/frozenproof, no independentdiskreload. Strata: {"SW104": {"default": 98.71563225845625, "continuous": 17.631889060480088, "overall": 74.12175282815869, "continuous_zero_fraction": 0.6661728845233483}, "SW252": {"default": 98.64493054166374, "continuous": 17.884868139910104, "overall": 74.14922820781278, "continuous_zero_fraction": 0.6639166799608013}, "ordinary": {"default": 98.65618782716787, "continuous": 17.770924133590587, "overall": 74.12251011928777, "continuous_zero_fraction": 0.6643952688074022}, "balanced": {"default": 98.7661680040883, "continuous": 17.586257646102915, "overall": 74.14311967451279, "continuous_zero_fraction": 0.6669021627657877}}.

# v246 additive conditional distribution versus logmass repair

## Focus
Allow small learned distribution to place probability outside strong prior modes. Additive mass=(1-alpha)*prior+alpha*softmax(newlogits), sigmoidalpha learned from frozen128hidden. Matched control uses samealpha uniform background plus bounded logmass correction. Same12609params/seed/halfhard-halftolerance CE/SAM.

## Explored and validation
Prepared/untrained, held until245 pair terminal. Original104/defaultgate frozen. Both start alpha.001 and uniform new component, hence equal initial distribution. Initial function differs from104: measure initial score separately, do not call deterministic mixture change training gain. All144665 actualassembly and restoredRAM/frozenchecks at finish.

## Gain
Unproved; paired additive-minus-control and absolute vs219 both required. Only one adoptedSWaddition/sixalone. Max2jobs; individual and pair totalwallclock recorded separately, model/data/initialeval included in trainerclock, preparation excluded.

## Not worth continuing
245 stillrunning, no final feature claim. Prior bounded logmass cannot arbitrarily move mass out of near-zero bins: mathematical limitation, frequency/importance on this dataset unmeasured. This is a hypothesis test, not a claim it is the empirically established bottleneck. No width grid or oldrepaircheckpoint chaining.

246 bothlive/initialeval complete: SW74.121701580675 equal acrossarms, vs104-0.000051247483 deterministic mixture/floating change, not traininggain. Both12609params/sameinitial. Newdistribution pair uses max2; owntrainedgain unproved. Originalbest models preserved.

V246 additive final SW74.121881069311, actualTOTAL80.612520302398; learned gain+0.000179488636, vs211-0.025741553316. Six optionalphases, default5; paired additive-vs-multiplicative distribution mechanism, identical initial mixture.

V246 multiplicative final SW74.149145254062, actualTOTAL80.622062767061; learned gain+0.027443673388, vs211+0.001522631435. Six optionalphases, default5; paired additive-vs-multiplicative distribution mechanism, identical initial mixture.

246 paired bounded final18both: additive74.121881069311 vsmultiplicative74.149145254062, gap-0.027264184751; newdistribution alone not useful here, no convergence/globalmechanism claim. ControlSW vs219+0.000965822451. Actualoptional7TOTAL80.624827357565, gain+0.000338037858. Both source/controlweights kept (additive247dependency); default5 unchanged. Concurrent pairwall toreport525.936s vsindividual519.04/518.34s; no serialspeedup claim. Next247aux+248PERMtransfer, original104/154only, oneaddition each,max2. No808/full82proof.

246 updated own validation diagnosis: continuous score 17.884647 vs104 17.631889, gain 0.252758; default score change -0.070724. Diagnostic only; not training weights/row selection or a gate classifier accuracy measure. Frozen gate does not prevent changes in final default action.

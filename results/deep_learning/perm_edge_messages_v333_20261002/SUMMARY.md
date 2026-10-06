# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM nonlinear local edge-message encoder v333

## Focus / control
Original154 frozen prior ONLY plus fresh312 positive sparse-selection encoder/rational/readout. Before first selector, add affinefreeLN48 center/neighbordelta concat96 -> Linear24/GELU messages from fixed offsets-3..3 -> max or maskedmean -> zeroLinear24to48 -> .25tanh bounded residual. 3528 new/40098 active, allfreshbody/readout/shapes/edgejoint onePERMphase/max8. Initialnewfunction samefresh312. Sameparameter/init/messagefunction onlyaggregation differs, max and mean effective function classes differ. Valid same-well rowneighbors, selftap present, invalid centerszero; no physical depthgap compensation, dynamicKNN/retrieval or fullDGCNN claim.

## Explored / validation
Hypothesis: center-conditioned cross-channel nonlinear neighbor messages improve continuous representations beyond point sparse selection. Different325recursiveEMA,190WTConv/nativewavelet,192linearDW/addmul,326/328perchannelSEmaxmin; notfirstlocalcontext/neuralmessageclass. Centerplusdelta versus centerplusneighbor is linearly interchangeable, so not a fake ablation. Existing206 derived context means no first access to local information. Zerooutput delays message gradients; maxwinner gradients sparse versus mean dense; may smooth rare/extreme values. No pertrialpreflight/review. Original64/16fixedfifth/seed20261010/Q50/9curves206relative/alpha.

 No pertrialpreflight/review. Original64/16fixedfifth/seed20261010/Q50/9curves206relative/alpha.3ramp5/80pat14/b64x64/cos80/decoder/ownwholePERMbest unchanged. No312repairweights/labels in neighborhoods.

## Gains
Prepared/untrained/no gain. Retained312 whole89.51580440487884/CONT74.00288840401272. Predefined max-minus-mean paired plus each absolute; own whole-target checkpoints. Pairedpositive belowbest is mechanism evidence only, adoption requires bothwhole andCONT improvement.

## Closure / costs
Only two namedarms, no neighbor/radius/messagewidth/LRgrid or queues. Record actualindividual/pair wall/concurrency/peaks, prep initial train restore included/implementation CPUfinalization excluded. No remaining folds below strict80.8 and gain. Default5/retained8TOTAL80.699037441 unchanged/fullOOF82 unmet. Prepared while332twoGPU live; do notlaunch until slotreleased.

333 LIVE: maximumPID59720/meanPID80892 after332terminal, GPUcap2. OrdinaryinitialPERM89.4055666268939/40098trainable both. Original154only+fresh312code+3528edge-message, allnewbody/readout/shapes/messages joint, onePERMphase/max8. No learnedgain yet/otherfolds. [Primarysource/history/interface research](../cross_domain_research_20261001/PERM_EDGE_MESSAGES_RESEARCH_20261002.md).

V333 final PERM89.499348140436, gain+0.093783283517, actualTOTAL80.645299567225; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds268.503931/peakMiB964.092773. No otherfolds.

V333 final PERM89.505470296241, gain+0.099905439322, actualTOTAL80.647442321757; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds266.493704/peakMiB946.983398. No otherfolds.

## Final closure
Both natural19/best5: maxPERM89.49934814043613/CONT74.00681575433087; mean89.50547029624123/CONT74.01617967397496. Maxminusmean whole -0.006122155805/CONT -0.009363919644. Versus312 maxwhole -0.016456264443/CONT +0.003927350318, meanwhole -0.010334108638/CONT +0.013291269962. Noadoption, close nonlinear-edge setting without radius/messagewidth/LRscan. Zerooutputactuallylearned normsmax1.631627/.187094 vsmean1.905923/.283305; all144665 RAMbest/frozenexact, notindependentdiskreload. Pair274.030683s/stagger1.119s, individuals268.503931/266.493704s, peaks964.093/946.983MiB, prepinitialtrainrestoreincluded/implementationfinalizationexcluded. Nominalsame80-14, effective max/meanfunction/gradient sparsity different, inheritednull27fields notapplicable. Retain8TOTAL80.699037441/default5/strict808false/nootherfolds/fullOOF82 unmet.



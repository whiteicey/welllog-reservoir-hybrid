# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V353 POR dynamic cross-spectrum transfer

## Focus
Transfer terminal-positive351 amplitude-only CrossSpectrum code/init to fresh341POR, adding552parameters after existing curve/depth correction at rawstem, before rawencoderbranches. Retain whole134752positive curve64+depth17tap+conditionalRPS structure andoriginal151prior. Input actual256PORwindow, rFFT512/ortho nineoriginalcurves, centeredvalid/constanteligibility, donorlogamp normalization/std floor.1/clip4,72offdiagsoftmax, .5tanh realmultiplicativegate/DC0/targetphase preserved, filtered-minus-original +zero9to48bridge .25tanh. No phase352module or trained351/341weights.

## Explored and verification
Prepared/nottrained/noGPU. Allfreshbody/readout/curve/depth/spectrum jointly train135304 in oneaddedPORphase/max8. Original151only. Sameparams/init aligned versus fixednonDCdonorpermutation, target/outputcoordinates notpermuted; controlalso changesdonorsmoothness/functionclass, notpurecausal. Zero bridge keepsfresh341initialfunction. Historical341134752POR78.046430717932/CONT41.736640291995/noextra552 capacity, nonconcurrent; reportoriginal151CONT41.953961054904 too. 351SWpairedpositivewhole+.004675855/CONT+.024578354 vspermuted, optional8TOTAL80.706129723238. Transfergainnotimpliedbysourcegain. StaticFourier/PORcrosscorrelation historicalnegative, no firstspectralclaim. Frequency rowgrid notphysicalirregularsampling, constan

portoriginal151CONT41.953961054904 too. 351SWpairedpositivewhole+.004675855/CONT+.024578354 vspermuted, optional8TOTAL80.706129723238. Transfergainnotimpliedbysourcegain. StaticFourier/PORcrosscorrelation historicalnegative, no firstspectralclaim. Frequency rowgrid notphysicalirregularsampling, constant0.1traininputs do not makeboundaries harmless. Existing206missing/relative bypass unchanged.

## Gain
Unproven forPOR. Originaldata/split/Q50/ownPORtruth/scoring80pat14/min10/b64x64/cos80 andofficialdecoder unchanged. Wholebestselect, DEFAULT/CONT samecheckpoint,42prefixrecorded tocomparehistorywithoutclaimingcapacitymatched. No CONToracle/pseudo/teacher/KL/externalvoting/newmeasurements. OneadditionalPORstage; sixwithdefaultotherheads/eightwithretainedPERM336+SW351. Olddefault5/current351eight/deliveries unchanged. StrictactualTOTAL>80.8+clear gain beforeotherfourfolds; fullOOF82unmet.

## Not worth extending
No frequencyband/window/width/LRgrid ifmatchedmechanismfails. 350paralleltemporalnegative preservedasclosed andnotincluded. This transferspositive351code ratherthanrevivingnegative350/usingrepairweights. Ifmatchedlocalpositivebutabsoluteweaker, retainhonestcodelesson/notnewbest; don'twritefull5foldbenefit.

## Timing and next
Prepared only while352phase/magnitude study continues; noGPUlaunch orautomaticqueue. Solo preferred duemeasuredcontention/max2cap. Recordindividualprep/warmup/train/restoration/peak andpairwall/staggerincludinggaps. RAMbestreplay notindependentdiskreload.


## Explicit launch and live snapshot
Aligned explicitly launched solo as PID30716 after both352arms terminal and globalGPU0. {"status": "running", "pid": 30716, "stage": "aligned", "epoch": 37, "best": 78.04357779977842, "best_epoch": 28, "bad": 9, "peak_MiB": 2222.54248046875}. Interim best is not final gain versus341 POR78.04643071793232. Permuted prepared/not launched/no automatic queue. Original151only/allfreshencoderreadoutjointonePORphase, training source/config unchanged.

V353 terminal POR78.04357779977842/CONT41.72843647938606, best/stop28/42, diagnostic vsserial345 whole-0.0017124389843985455/CONT-0.0088997770868815, vs341whole-0.002852918153905648. Diagnostic8TOTAL80.7052738477919; adopted=False. POR351 amplitude-spectrum transfer below retained341whole; permuted not trained/no matchedgain claim, close transfer configuration no grid; SW351positive preserved. Permutedarm notlaunched/no matchedgainclaim. Solo individual523.3101248000166s/peak2222.54248046875MiB includesprep/warmup/restoration;144665RAMbestreplay/frozenexact/notdiskreload/fullOOF82unmet/nootherfolds.



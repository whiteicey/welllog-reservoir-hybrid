# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# Conditional probability chain v311

## Focus
CORN-inspired probability factorization with fresh297well encoder, state/FiLM/body and density readout jointly trained in one SWphase; only original104 prior loaded. Replace flat-softmax readout by255 conditional survival probabilities derived from the original256 linear outputs: a_j=ell_j-ell_255. All256 rows active, common-shift redundancy; same65265 stored trainable parameters. No extra ordinal loss, teacher or chain of trained repairs.

## Fair contrast and validation
Ordered chain follows original physical bin order; shuffled chain uses one fixed seed20261311 permutation and inverse mapping before all loss, mixtures and decoder. Both same new weights0, bias_j=log(255-j), bias_255=0, hence identical uniform new q and original104 complete initial function. First step body density gradient zero is expected; subsequent updates must jointly train actual body/FiLM/readout/state. Existing second-step group-gradient/finite/frozen/population/best-RAM checks remain. Different conditional task populations and finite function classes; matched storage is not proof of same effective capacity.
FP32 logsigmoid and exclusive cumsum create logmass; no BF16 long products or adjacent-CDF subtraction. Direct halfhard/physicalsoft CE1 over existing four feature outputs, mixedCE/conditionalRPS1/balancedBCE/.25ramp5/statebound4 unchanged. Original bi

s/controls.

## Sources
[Research including history](../cross_domain_research_20261001/SW_CONDITIONAL_ORDINAL_RESEARCH_20261002.md).
[CORN primary paper](https://arxiv.org/abs/2111.08851): conditional probability chain removes CORAL output weight-sharing constraint. Our use is a migration inference; paper does not establish well-log or256-bin gains.

## Timing
Record individual time/peak allocation and same-work two-process wall including data/initial/training/best restore; exclude codeprep/CPUrecord. No automatic launch or serial-speed claims.

V311 ordered explicitly launched PID90368 while310channel final epochs occupy other slot; shuffled not launched yet, no automatic queue. First real gradients/init/results pending.

Explicit311 pair now running: ordered PID90368/shuffled PID82616, each65265jointparams, original104 initialSW74.121752689769 exact. Source/activeconfigs frozen; gains pending final ownwhole-best.

V311 running snapshot, not final gain: {"ordered": {"status": "running", "pid": 90368, "stage": "ordered", "epoch": 9, "best": 74.14793432961145, "best_epoch": 6, "bad": 3, "peak_MiB": 1484.1552734375}, "shuffled": {"status": "running", "pid": 82616, "stage": "shuffled", "epoch": 4, "best": 74.13525951818056, "best_epoch": 4, "bad": 0, "peak_MiB": 1484.1552734375}}; real source initialization74.121752689769/65265params, actual joint training passed gradient/frozen checks. Retained8unchanged; remainingfolds not started.

V311 ordered: SW74.147934329611, actualTOTAL80.621638943503; fresh65265 joint encoder/conditional-chain readout, original104only; oneSWphase, nofullOOF/remainingfolds.

V311 shuffled: SW74.149063522486, actualTOTAL80.622034161009; fresh65265 joint encoder/conditional-chain readout, original104only; oneSWphase, nofullOOF/remainingfolds.

Final311 whole scores {'ordered': 74.14793432961145, 'shuffled': 74.14906352248619}; ordered-shuffled -0.001129192875; CONT paired -0.086347350318; strata {'ordered': {'default': 98.87157026578004, 'continuous': 17.360031882823368}, 'shuffled': {'default': 98.8355981992487, 'continuous': 17.44637923314177}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 689.697s; no fullOOF or remaining folds.

V311 closed: ordered74.147934329611/shuffled74.149063522486, pairedwhole-.001129192875/CONT-.086347350318; both below297 and252. Noadoption or chain/bias/lossgrid. FullRAM/frozen144665replay, actual6composition, best/stop6/20 and17/31, wall689.697s includingstagger/variableoverlap; negative scope not all ordinalmethods.



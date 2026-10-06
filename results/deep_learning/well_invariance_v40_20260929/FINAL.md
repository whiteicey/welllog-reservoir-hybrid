# v40 Training-well gradient reversal: final results

Original fifth TOTAL 80.336513436816 -> 80.336513436816, delta +0.000000000000.

|Gradient|Trainable parameters|Best SW after update|SW with parent fallback|Best/stop epoch|
|---|---:|---:|---:|---|
|blocked|25728|73.645195910|73.663082797|0/18|
|reversed|25728|73.645195910|73.663082797|0/18|

## Mechanism and verification
Matched5120 existing SW normalization affine parameters and20608 auxiliary classifier parameters. Both classifiers minimize train-well CE; only reversed sends the negative gradient to the backbone, after the prespecified warmup/ramp. Original heads/convolutions/routes, data/Q50 and decoder stay fixed.
Historical well_adversarial_sequence was a multi-target full-network single-classifier study. This is a current-private-SW warm-start comparison, not its restart or a first discovery of gradient reversal.
Well IDs are auxiliary labels for64 training wells only. Held-out wells map to-1 and never enter auxiliary training. IDs and classifier outputs are absent from prediction inputs. First actual batch verified trainable gradients; final frozen-state and144665 selected prediction restoration checks passed.
Selected SW full disk reconstruction: False. Fixed three-column scoring completed.

## Classifier diagnostics and comparison
Identical warmup SW score differences: {"1": 0.0, "2": 0.0, "3": 0.0}
blocked final training classifier diagnostics: {"domain_cross_entropy": 3.3249366469681263, "domain_training_accuracy": 0.1650390625, "grl_coefficient": 0.0}; majority-class prior=0.022486; normalization change L2=1.430216.
reversed final training classifier diagnostics: {"domain_cross_entropy": 3.337776582688093, "domain_training_accuracy": 0.162841796875, "grl_coefficient": 1.0}; majority-class prior=0.022486; normalization change L2=1.396676.
Reversed-blocked selected SW difference +0.000000000000, post-update-only difference +0.000000000000. Adopted=None.
A learnable training-well classifier is not causal proof of harmful domain dependence. Positive differences remain selected development-fold evidence; they are not independent generalization proof.
Close this coefficient/curriculum setting if neither arm improves the parent. Do not conclude that all adversarial learning is ineffective. Epoch0 fallback is not training gain.

## Timing and outstanding objective
Same-work wall time 196.78s, measured peak process overlap=2; individual times and peak memory are in timing.json.
CUDA linear interpolation backward uses warn_only and is not claimed bitwise repeatable. Original5foldOOF>82 remains unachieved. No other folds, old jobs or delivery replacements were started.


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 before reversal starts. For the prescribed active epochs4..18, blocked best73.5954025454393 and reversed best73.59465943798016 (delta-0.000743107459). Final SW difference-0.006539743169. This diagnostic does not change checkpoint selection. Final domain accuracy blocked0.165039/reversed0.162842 vs majority prior0.022486: classifier learns, but the reversed objective does not improve this SW adaptation setting.
Do not extend this adversarial coefficient/curriculum scan. Next pending direction: inspect PERM conditional-density family/loss alternatives against the current retained PERM89.396724 model; preserve original actions/scoring and distinguish an initial distribution intervention from a retraining benefit. No new PERM run implemented or launched yet. Original fullOOF>82 and fifth>80.5 remain outstanding.

# SW coarse-to-fine continuous density v299

## Focus
New raw encoder jointly learns broad continuous regime and within-regime detail. Eight contiguous groups of the EXISTING256bins, training64wells only/Q-weighted continuous-label quantiles. Quantile cuts clamp to nonempty groups; no labels/IDs as inference features, no changed actions or groupwise oracle. Fresh48rawbody inherited289code plus49statehead, original104 prior frozen; no289/297trained weights. Protected .25mix/ramp5, statebound4/balancedBCE/directcontinuousCE1/conditionalRPS unchanged.

The shared48features receive per-group learnable48gamma/48beta FiLM bounded.25 (zero init). Shared fine256 linear outputs, coarse8 linear outputs. Hierarchical q_j=softmax(c)_group*softmax(v_within_group)_j. Flat control q=softmax(c_group+v_j-log(group_size)). Same body, registered and active parameters, complete initial tensors; change only normalization. Coarse initialweights0/biaslog(group_size), fineweights/bias0 -> uniform q in BOTH; alpha0 original104 prediction. This initialization differs289 randomreadout, so historical289 is absolute control, not strict architecture ablation. At arbitrary free logits both express a full categorical distribution; no claim hierarchy adds information or universally greater capacity. Its hypothesis is separating coarse mass competition from fine gradients, with identical group-specific feature processing in both arms.

## Explored and validation
Prepared/not launched. Ordinary initialeval checks sourcefunction, actual allnew groups trained, atomic laststate/best and groupboundaries. Boundaries saved as buffer; restored Python slices synchronized on load. Directcomponent auxiliary uses same normalized q for branch and finalfeature reads, no synthetic label signals. Each ownbest selected by wholeSW, default/continuous scores diagnostic only. Actual144665 composition after terminal; RAM replay not independent diskreload.

## Gain
Unproved. Primary hierarchical-minus-flat final wholeSW AND continuous stratum; absolute28974.1537264289993 and252continuous17.88486813991017. Preserve retained positive models/candidate29780.6935127, never adopt a worse replacement. One newSWphase from104, max8 wholemethod; strict80.8/usefulgain before remainingfolds, OOF82 unmet.

## Unworthy followup / failure conditions
If only default improves or hierarchy fails flat/retainedcontinuous, stop this8group setting, no groupcount/quantile/FiLMwidth grids. Groups can separate training modes that do not generalize across wells. Narrow groups may impede flow of softtolerance supervision across boundaries; counts may concentrate into a few bins, clamp disclosed. Original rawbody stillpointwise; hierarchy cannot restore absent input information. This is cross-domain inspired, not HCA teacher/range-preserving distillation or full original-paper reproduction.

## Timing
Individual data/initial/train/bestRAMrestore wall and peakMiB; same-work pairedwall with actual overlap/gaps; codeprep/CPUassembly excluded. Max2GPU only after authoritative admission, no redundant preflight.
Implementation transfer choice: v299 fits Q-weighted CDF cuts via first cumulative bin reaching each quantile +1 then nonempty clamp, not the research sketch nearest-count boundary. Both arms actualfit rows161423 and boundaries0,202,219,228,234,239,244,249,256 agree; activeparams58777 each. CUDA bincount warns nondeterministic floating accumulation, but actual integer boundaries match; preserved source unchanged. Research source: [CVPR2024 HCA](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Deep_Imbalanced_Regression_via_Hierarchical_Classification_Adjustment_CVPR_2024_paper.html) and [authors code](https://github.com/xhp-hust-2018-2011/HCA); no teacher/detach/range-preserving distillation, not full paper reproduction. See cross_domain_research_20261001/SW_HIERARCHICAL_CONTINUOUS_RESEARCH_20261002.md.

V299 flat final SW74.153482146373, actualTOTAL80.623580679369; learned gain+0.031729456604, vs211+0.005859523746. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V299 hierarchical final SW74.155778635294, actualTOTAL80.624384450492; learned gain+0.034025945525, vs211+0.008156012667. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final matched result
Flat SW74.153482146373, hierarchical74.155778635294; paired overall+0.002296488921, continuous flat17.382210710024/hierarchical17.404116921379, pairedcontinuous+0.021906211355. Hierarchical-minus297-0.004251587726, not adopted. Compare absolutecontinuous25217.88486813991017; no continuous breakthrough from a local relative benefit alone. Same actual8boundaries/fit161423rows/58777params and sourceinitialfunction, allnewencoder/readout/state grads verified/144665composition/RAMbest replay, not independentdiskreload. Individual flat374.755s/hierarchical371.262s, pairwall420.451s, partial overlap with otherjobs, no speedranking. Do not sweep groupcount/quantile/headwidth on thissmall overall effect. No additionalfolds, preserve current optional8/default5/oldfullOOFdelivery.

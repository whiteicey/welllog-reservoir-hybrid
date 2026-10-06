# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v40 训练井不变性：阻断与反转梯度

## 探索重点与历史区别
固定80.336513436816基线，针对v39单纯SW归一化微调无收益，检验训练井特征依赖是否妨碍泛化。旧well_adversarial_sequence是多目标从头训练、单128维线性分类器、系数至.01，原第五79.843176；本次是强SW双专家热启动、仅5120个归一化仿射参数更新，配备同一个256→64→64训练井分类器。不是首次尝试井对抗思想，也不重复旧任务。

## 预定义匹配对照
blocked/reversed两个独立进程，同完整初始化、同NLL/Q50/采样窗口/优化器/参数集合。两组都训练井分类器；blocked井分类梯度不进骨干，reversed在前三轮也阻断，第4至8轮将反转系数从.2升到1，之后固定1。总损失SW原NLL/3 + .1*井分类CE。分类器正常最小化CE，骨干在reversed中收到相反方向的井分类梯度。
两个专家主读出前128维隐藏特征，各自按有效中心128点平均，拼成256维；训练井ID仅用作分类监督，不作为预测输入。只映射64训练井，16验证井的域标签固定为-1，训练锚点严格检查有效映射。验证不会运行或更新分类器。无验证标签/井参与辅助拟合、无跨井输入窗口。
原5120个GroupNorm/LayerNorm缩放偏置更新，旧卷积/头/路由固定。新增20608参数仅属于训练井分类器，推理不用其输出；两个候选总可训练参数相同。真实梯度和归一化权重变化记录，避免把分类器单独学会当作骨干有效训练。

## 训练与验证边界
原9曲线/206输入与63既有幅值侧特征、原第五折144665点评分/动作不改。max80/pat18/min10/b64/64步；norm AdamWlr1e-4/wd0，分类器lr3e-4/wd.01；cosine80*64至1e-5，两个参数组各clip2。两进程并发上限2，无自动队列/恢复。
初始模型等于原SW73.663082797066；独立按SW选最佳，完成后恢复并和固定POR/PERM实际组装。不做重复预检/独立审查。训练侧分类正确率仅作诊断，需与训练井最大先验类比例比较，不作为预测评分。

## 已完成、未完成与后续
已实现两候选；训练收益待定。若井分类器不能学习，本次只能说明该训练设置信号不足；若能学习但反转无收益，则不继续该对抗系数扫描。不将训练井可区分性本身当作泛化失败的因果证据。
完整OOF>82尚未达成，第五严格>80.5并有明显收益才考虑其余折；本批无其他折/旧任务/交付覆盖。每批维护PROJECTS.md、README、EXPLORATIONS和图谱，保留真实失败与负结果，删除无用checkpoint。


## v40骨干归一化对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见well_invariance_v40_20260929/FINAL.md。


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 bef

# v40 训练井不变性：阻断与反转梯度

## 探索重点与历史区别
固定80.336513436816基线，针对v39单纯SW归一化微调无收益，检验训练井特征依赖是否妨碍泛化。旧well_adversarial_sequence是多目标从头训练、单128维线性分类器、系数至.01，原第五79.843176；本次是强SW双专家热启动、仅5120个归一化仿射参数更新，配备同一个256→64→64训练井分类器。不是首次尝试井对抗思想，也不重复旧任务。

## 预定义匹配对照
blocked/reversed两个独立进程，同完整初始化、同NLL/Q50/采样窗口/优化器/参数集合。两组都训练井分类器；blocked井分类梯度不进骨干，reversed在前三轮也阻断，第4至8轮将反转系数从.2升到1，之后固定1。总损失SW原NLL/3 + .1*井分类CE。分类器正常最小化CE，骨干在reversed中收到相反方向的井分类梯度。
两个专家主读出前128维隐藏特征，各自按有效中心128点平均，拼成256维；训练井ID仅用作分类监督，不作为预测输入。只映射64训练井，16验证井的域标签固定为-1，训练锚点严格检查有效映射。验证不会运行或更新分类器。无验证标签/井参与辅助拟合、无跨井输入窗口。
原5120个GroupNorm/LayerNorm缩放偏置更新，旧卷积/头/路由固定。新增20608参数仅属于训练井分类器，推理不用其输出；两个候选总可训练参数相同。真实梯度和归一化权重变化记录，避免把分类器单独学会当作骨干有效训练。

## 训练与验证边界
原9曲线/206输入与63既有幅值侧特征、原第五折144665点评分/动作不改。max80/pat18/min10/b64/64步；norm AdamWlr1e-4/wd0，分类器lr3e-4/wd.01；cosine80*64至1e-5，两个参数组各clip2。两进程并发上限2，无自动队列/恢复。
初始模型等于原SW73.663082797066；独立按SW选最佳，完成后恢复并和固定POR/PERM实际组装。不做重复预检/独立审查。训练侧分类正确率仅作诊断，需与训练井最大先验类比例比较，不作为预测评分。

## 已完成、未完成与后续
已实现两候选；训练收益待定。若井分类器不能学习，本次只能说明该训练设置信号不足；若能学习但反转无收益，则不继续该对抗系数扫描。不将训练井可区分性本身当作泛化失败的因果证据。
完整OOF>82尚未达成，第五严格>80.5并有明显收益才考虑其余折；本批无其他折/旧任务/交付覆盖。每批维护PROJECTS.md、README、EXPLORATIONS和图谱，保留真实失败与负结果，删除无用checkpoint。


## v40骨干归一化对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见well_invariance_v40_20260929/FINAL.md。


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 before reversal starts. For the prescribed active epochs4..18, blocked best73.5954025454393 and reversed best73.59465943798016 (delta-0.000743107459). Final SW difference-0.006539743169. This diagnostic does not change checkpoint selection. Final domain accuracy blocked0.165039/reversed0.162842 vs majority prior0.022486: classifier learns, but the reversed objective does not improve this SW adaptation setting.
Do not extend this adversarial coefficient/curriculum scan. Next pending direction: inspect PERM conditional-density family/loss alternatives against the current retained PERM89.396724 model; preserve original actions/scoring and distinguish an initial distribution intervention from a retraining benefit. No new PERM run implemented or launched yet. Original fullOOF>82 and fifth>80.5 remain outstanding.


## FINAL.md

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
blocked final 

ssifier outputs are absent from prediction inputs. First actual batch verified trainable gradients; final frozen-state and144665 selected prediction restoration checks passed.
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



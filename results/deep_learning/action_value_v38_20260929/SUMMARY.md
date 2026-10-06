# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v38 直接动作价值读出

## 目标与假设
基于已验证原第五折TOTAL80.336513436816，改为直接学习每个候选预测动作的期望官方得分，检验密度似然代理与最终动作决策的差异。
旧v12在独立TCN密度模型上添加softmax动作效用项，最佳回退SW73.093513；本次保留当前更强的预测器，新增直接动作价值回归头，不沿用该密度损失。推理仍为原动作集合argmax。

## 预定义对照
POR/SW各point、local5两项，共四项；PERM固定。319输入经32维GELU再输出原全部动作的奖励修正，界限±0.25。两候选同参数/同初始化/同训练窗口，唯一结构差异是32维隐藏读出前是否应用同井5点有效掩码平均池化。新增参数为10240+33×实际动作数，旧预测器全部冻结。初始输出为零，与原密度期望效用一致。
每个训练点的监督是其真实标签对所有动作的官方kernel得分，不是验证标签、不选最优动作伪标签。均方误差学习条件期望，所有动作和默认动作均包含；平均后除3，无NLL混合。两候选统一等权，承接v37低于基线但相对Q50的小幅匹配收益，不能视为已获当前绝对收益。
输出是动作奖励估计而非概率；不要求各动作和为1。动作数/动作值/评分公式不改；训练labels逆变换依旧用原训练侧median/scale。

## 已完成与待完成
已实现新模块和独立训练器；待实际训练、最佳恢复、每目标比较与三列组装评分。收益尚未证实。
原9曲线、206主特征、63已有幅值侧特征、原64/16井及144665验证点保持；训练权重为全1。80轮上限/patience18/min10/batch64/64步，AdamW3e-4/weight_decay.01，cosine80×64至1e-5，clip2。
最多2目标GPU进程，每进程2候选独立参数/优化器/早停，共享冻结前向。原子last_state带完整优化/RNG状态；无自动续跑。

## 不延伸与后续
不继续v35/v36检索扫参及v37全因子NLL头扫参。若本次不同读出仍无收益，记录局部范围结论并换方向；不把一个负实验扩展为直接效用方法普遍无效。
严格第五>80.5并有明显收益才考虑原其他折；完整OOF>82未完成。无旧任务启动、其他折或交付覆盖。维护PROJECTS.md、README、EXPLORATIONS与图谱，区分未训练/进行中/最终成绩。


## 最终报告
见[FINAL.md](FINAL.md)，TOTAL=80.336513436816。




## FINAL.md

# v38 Direct official action-value readouts

Fixed original fifth TOTAL 80.336513436816 -> 80.336513436816, delta +0.000000000000.

|Target|Context|Best after update|Selected with parent fallback|Best/stop epoch|
|---|---|---:|---:|---|
|POR|point|77.507769906|77.551936802|0/18|
|POR|local5|77.508897729|77.551936802|0/18|
|SW|point|73.614883759|73.663082797|0/18|
|SW|local5|73.605314096|73.663082797|0/18|

## Scope, execution correction and verification
Same initialized parameters, original input/hidden features/actions/score; identity vs five-point masked average at the hidden readout. All parent predictors frozen. New heads regress official action utility with uniform squared error, not a likelihood or soft-action objective.
The first POR run incorrectly inverted log1p with exp. It was stopped after epoch4; its scores are excluded and no weights were reused. Corrected POR restarted from the original parent, uses expm1, and checked 2048 training labels against the official kernel (maximum difference below1e-4). SW log/exp was correct and finished unchanged.
Selected-head full disk reconstruction: []. Fixed three-column predictions scored on144665 original fifth rows.

## Mechanism contrasts and scope of conclusions
POR local-point selected channel difference +0.000000000000; post-update-only difference +0.001127823503.
SW local-point selected channel difference +0.000000000000

# v38 Direct official action-value readouts

Fixed original fifth TOTAL 80.336513436816 -> 80.336513436816, delta +0.000000000000.

|Target|Context|Best after update|Selected with parent fallback|Best/stop epoch|
|---|---|---:|---:|---|
|POR|point|77.507769906|77.551936802|0/18|
|POR|local5|77.508897729|77.551936802|0/18|
|SW|point|73.614883759|73.663082797|0/18|
|SW|local5|73.605314096|73.663082797|0/18|

## Scope, execution correction and verification
Same initialized parameters, original input/hidden features/actions/score; identity vs five-point masked average at the hidden readout. All parent predictors frozen. New heads regress official action utility with uniform squared error, not a likelihood or soft-action objective.
The first POR run incorrectly inverted log1p with exp. It was stopped after epoch4; its scores are excluded and no weights were reused. Corrected POR restarted from the original parent, uses expm1, and checked 2048 training labels against the official kernel (maximum difference below1e-4). SW log/exp was correct and finished unchanged.
Selected-head full disk reconstruction: []. Fixed three-column predictions scored on144665 original fifth rows.

## Mechanism contrasts and scope of conclusions
POR local-point selected channel difference +0.000000000000; post-update-only difference +0.001127823503.
SW local-point selected channel difference +0.000000000000; post-update-only difference -0.009569662511.
Adopted: []
Epoch0 fallback is not training improvement. Positive comparisons are development-fold evidence, not independent validation. Both valid targets failed to improve the parent; close this direct-value readout setting without an additional width/context/loss sweep.

## Time and outstanding objective
Valid training interval wall time 354.385796 seconds; per-target times, peak memory and invalid-attempt overhead are in timing.json.
Full original5foldOOF>82 is unachieved. No other folds, old tasks or delivery replacements were started.



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


## v38 SW 显式启动
{"pid": 55412, "target": "SW", "started_at": "2026-09-29T14:45:48.281608+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_SW_fold5"], "automatic_queue": false}


## v38 POR 显式启动
{"pid": 14968, "target": "POR", "started_at": "2026-09-29T14:46:06.946657+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_POR_fold5"], "automatic_queue": false}


## Invalid POR objective stopped
Wrong inverse exp instead of expm1 for POR log1p labels. PID14968 stopped after epoch4; all recorded gains are invalid for method attribution and excluded from selection. Historical source/config/checkpoints not mutated. New corrected run restarts from the original baseline, not this state. SW uses log and exp correctly and continues unchanged.


## v38 corrected POR 显式启动
{"pid": 23728, "target": "POR", "started_at": "2026-09-29T14:49:07.153015+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38_por_fixed.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_POR_fixed_fold5"], "automatic_queue": false}


## v38直接动作奖励对照完成
四项完成，TOTAL=80.336513436816，相对80.336513增加+0.000000000000；采用[]。完整记录action_value_v38_20260929/FINAL.md；错误POR初次运行被剔除，无其他折或交付改动。


## v38 closure and retention
All four valid candidates stopped after18 non-improving epochs and restored epoch0 exactly. Keep baseline TOTAL80.336513436816. Original POR wrong-inverse run excluded; corrected2048-point training-kernel max error4.8336e-6, SW also checked at finalization. Original launcher now only admits SW; POR has a separate corrected launcher. No selected replacement.
15 negative/invalid checkpoints totaling30001900 bytes deleted after final verification, SHA256/size manifest saved. Compact source/config/log/score/prediction and failure evidence retained, all baseline checkpoints preserved. No live training, automatic queue, extra folds or delivery changes. FullOOF>82 remains unachieved.
Do not expand the v37/v38 frozen output-head sweeps. Next work should assess a different update location inside the retained private backbone, or measure training-vs-validation feature behavior to justify it; this is a pending hypothesis, not an implemented or launched experiment.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

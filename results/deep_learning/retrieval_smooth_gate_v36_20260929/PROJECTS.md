# v36 进度与固定方案

## 目标
保留80.336513原模型，修复v35检索混合门控的负半轴失活，完成有效matched/shuffled对照。

## 已完成
v35已完成但新分支失活：7973个固定训练探针点的末态proposal全部负、alpha和梯度为0。保存gate_diagnostic.json，历史训练结果不改。v36模块和独立训练器已实现，复用合法v35参考库与索引，不重做检索准备。

## 本批设计
只把clamp(u,0,.25)改为.25*sigmoid(u+logit(.04))，初始参与比例1%。记录真实初始候选分，单独关闭检索校验父预测。查询/键/输出层需要非零梯度证据；训练后候选最高分与回退父模型分分别记录。两个候选均11266参数，相同初始化/库/索引/带宽/数据/损失/80-18预算，仅值与键对应不同。严格整口查询井排除，shuffled不部署。只在matched超过基线与shuffled时采用。

## 待完成
执行两项训练、最佳候选恢复、默认/数值段和参与比例诊断、实际三列评分。新采用模型才额外磁盘重建。未取得新分数前保持原配方。

## 后续与约束
不扫描初始参与比例、K、库规模、带宽或学习率。原其他折/旧任务/交付不改。维护独立README、根PROJECTS/EXPLORATIONS与图谱；区分v35失活与v36有效实验结果。无自动排队或续跑。


## v36显式启动
{"pid": 67396, "started_at": "2026-09-29T14:22:06.320868+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_retrieval_smooth_v36.py", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_retrieval_smooth_v36_SW_fold5"], "automatic_queue": false}


## v36检索门控对照完成
两项已完成，TOTAL=80.336513436816，增量=+0.000000000000；采用=None。v35失活证据和v36有效训练区别已记录。所有旧模型/交付保留，无其他折；见retrieval_smooth_gate_v36_20260929/FINAL.md。

Completed non-adopted checkpoints and neighbor index caches deleted after final recovery checks. Compact evidence retained; deletion manifests in v36 and v37 studies. Indexes require prepare_retrieval_v35.py to rebuild, not an available-cache claim.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

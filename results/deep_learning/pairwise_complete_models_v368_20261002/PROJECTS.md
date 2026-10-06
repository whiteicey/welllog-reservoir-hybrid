# 目标

六对已有完整模型按每头固定分布权重联合预测，验证是否严格超过80.74，同时保持训练继续。

# 已完成

- 固定四个完整模型和六个配对；旧首对101权重缓存可复用。
- 固定相同第五折、0.01权重网格、每头整通道选择与16历史阶段诊断边界。

# 计划与未完成

- 准备并执行共享原生前向与五个新配对评估，不新增训练。
- CPU评分、实际三头组合、默认/连续分解与六对结果表。
- 更新README/EXPLORATIONS/图谱与实际并发耗时。
- 原完整OOF>82尚未达成；本诊断不触发补折或自动采用。

# 后续可选

若某对有互补性，分析对应头与错误类型，再决定如何将有效机制带回八阶段方法。

范围增补：seed20261013 在评估期间完成八阶段，TOTAL80.3213137271093。保留最初6对，新增它与原模型/seed10/11/12的4对，共5个完整模型10对；seed14仍训练中，排除。无新训练阶段。

Completed 10-pair native distribution diagnostic over 5 completed models / 144665 original fifth-fold rows. Best original__seed_20261013 TOTAL=80.863457557382; delta80.74=+0.123457557382; delta prior80.735166=+0.128291319656. All new native model predictions and grid endpoints exact, each row covered once; first pair reuses verified cache. Fixed whole-head ratios, grid0.01; no row/CONT-specific selection. No new training stages, 16 preexisting stages per pair, no other folds/adoption/fullOOF claim. Training continues. Timing concurrent GPU sum 246.635s, CPU37.745s, studywall 657.116s including implementation/wait; no solo throughput claim.

V368 completed10pair diagnostic TOTAL80.863457557382/original+seed13, equal50:50=80.840351605081, >80.74 +0.123457557382. Fixed per-head weights56/44,44/56,68/32. CONTdeltaoriginal POR-0.248553617806/PERM+0.168665049892/SW-0.480118282476; not all-continuous improvement. No new training/fullOOF/adoption/otherfolds. Atlas updated with diagnostic scope; fourth new complete, last seed14 continues.

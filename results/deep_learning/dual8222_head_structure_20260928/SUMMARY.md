# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# 双专家80.22独立头：首轮结果

两组均完成，初始函数与原模型逐元素一致，最终三个最佳头共同恢复通过。仅POR采用第1轮，PERM和SW保留第0轮。

|结构|第五折TOTAL|相对80.221861|状态|
|---|---:|---:|---|
|pure_dl_dual8222_heads_pointwise80_fold5|80.25333950575663|0.03147842271695822|共同恢复已验证|
|pure_dl_dual8222_heads_local580_fold5|80.25447676690142|0.03261568386174929|共同恢复已验证|

局部头仅比逐点头高0.001137261145，不能认定其稳定更优。两组均93420可训练参数，但隐藏宽度80与77不同，是等预算结构比较。
本轮未增加仅复制线性头的匹配训练对照，因此约0.0326增益不能全部归功于非线性结构。下一步优先补该对照，再决定POR专用读出方向；不盲目继续扫核长和宽度。
两个结果均未超过80.5门槛，未补折、未替换交付。完整OOF最佳81.713233，目标>82仍未实现。旧ABD路线仍暂停。
last_state已保存当轮参数、优化器、调度器和随机状态，未进行中断续跑等价性测试。
本次图谱已生成。原curate.py的编辑被工具保护拒绝，原文件未改；新增说明在curate_dual8222_heads.py。以后运行主curate.py后须执行此扩展，再运行build_atlas.py。

详细证据见snapshot.json及两实验目录的fold_5_report.json。




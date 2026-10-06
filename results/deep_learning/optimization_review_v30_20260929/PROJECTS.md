# 优化方向诊断：已完成

## 目标
在保留80.336211945752当前模型的前提下，寻找还有证据支持的优化点；本轮只诊断与规划，不训练。

## 已完成
读取当前配方、损失、质量权重生成与特征实现。核算已有第五折预测的分层失分；按64训练井各4个固定分位窗口检查当前路由饱和，每目标32768训练点。保存diagnostic.json、可复现diagnose.py与README.md对照计划。

## 发现
POR连续路由88.47%的采样点触及95%以上修正幅度；SW非默认段67.51%零分，即使输出连续值仍有60.51%零分。Q50仅对1755/578891点降权，深度步长约0.1米均匀，现有物理代理与普通单曲线统计已较丰富。

## 计划与未完成
建议有限验证POR同边界饱和重参数化、SW低阶单调条件分布修正，随后检验跨曲线局部相关特征及交互头。输入边界统计和误差分层采样仍需额外诊断。上述模型改法、训练收益、独立种子和其他折都未执行，不启动自动任务。

## 目标达成后
固定已确认机制再按用户授权验证真正未见井；目前训练过的井不能重新划为独立验证而沿用旧权重。原模型、旧源码、数据与交付保持不变。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

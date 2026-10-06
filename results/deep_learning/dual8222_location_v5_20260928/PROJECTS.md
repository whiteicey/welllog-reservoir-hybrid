# 当前进度
目标：验证位置读出结构增量，完整OOF>82未完成。
已做：来源与三项对照已锁定，模型和训练入口实现。
计划：三组正式训练、完整第五折最终恢复、逐井及误差人口分解。
未做：真实训练收益、序列化重载验证；续跑等价性未实测。
达成后：固定有效方案再做泛化确认，当前不自动补折。


## 2026-09-28T10:06:43.427829+08:00 POR位置读出v5
三项预定义匹配实验已实现；本次实际启动[('linear', 49964), ('point32', 24972)]。初始、梯度和收益以实际日志为准，未自动启动其余队列/旧路线/补折。完整OOF>82未完成，旧交付不变。详细记录results/deep_learning/dual8222_location_v5_20260928。

## 本批最新结果
已完成：代码、明确启动、实际状态记录；全部完成=True，已验证保存重载=True。
当前最佳={'mode': 'point32', 'TOTAL': 80.27413533779057, 'delta': 0.005400828938988411}。
剩余：独立泛化及中断续跑等价性尚未验证；本批不再增加候选。
目标：第五折>80.5且明确收益，完整OOF>82未完成。达成后固定方案做泛化确认，当前无补折授权。



## 最终判断与保留配方
三项训练均完成且自然停止于新增第24轮，最佳均为第6轮。逐点头80.274135337791，比当前线性继续训练对照高0.002801716904；局部头80.273788310855，没有继续增益。全部保留原评分/骨干与其他通道。
已从readout_checkpoint.pt独立创建模型，完整144665点预测逐元素复现。推理时仅移除新增残差，分数80.273254588259，下降0.000880749532；这是训练后干预，不是匹配重训效应，也未作为第四个候选。
验证窗口的新增位置修正平均约0.004242倍固定标准差、最大0.126500倍，未接近0.25上限；该统计包含有效上下文位置，不能等同于仅中央带标签评分点。不能把保护上限本身归因为收益，因为没有无约束对照。
保留point32为本轮最高分开发候选。与线性对照比较12/16井提高、4井下降；探索性整井重采样区间未作反复选模校正，不能宣称稳定独立泛化优势。完整OOF>82与第五折>80.5仍未达成。
下一步尚未启动：优先检验训练目标与原官方容差效用的对齐，使用当前位置读出作固定控制；不继续无界扩大宽度/卷积/上限。旧ABD和其余折不恢复，交付不替换。
本轮配方见CURRENT_RECIPE.json，完整模型模块为dual8222_location_readout_v5.py。非线性读出不能仅按旧线性矩阵加载。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

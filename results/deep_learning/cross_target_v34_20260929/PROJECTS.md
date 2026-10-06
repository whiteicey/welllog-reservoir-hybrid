# v34 跨目标协同：已完成

## 目标
检验井外POR/PERM预测描述是否能在当前80.336513模型上改善SW。主预测器及POR/PERM输出不变。

## 已完成
完成4组内部交叉拟合，每组40井拟合、8井辅助NLL早停、16井接收OOF预测。POR/PERM/SW各自9,881参数MLP，总计12个模型均已训练和最佳恢复。所有辅助预处理仅用相应40井，查询井同时排除于拟合和早停，原第五折完全不进入辅助训练。生成完整730,268行预测描述及来源清单。
已完成self/cross/shuffled三个11,152参数的SW权重修正头，各自独立优化和早停，均第18轮停止并选择第0轮。更新后最高SW分别73.658076471、73.656148110、73.656649624，原SW为73.663082797。未选择任何新头，当前实际TOTAL80.33651343681649保持不变。冻结状态、初始一致性、训练内最佳恢复及实际三列评分均已核对；无新采用模型，未重复未变化模型的全量磁盘推理。

## 结论与停止方向
本次跨目标描述未超过当前模型，也未超过自身信息或打乱对应的负对照，不扩大这个小型辅助来源、六维描述及固定权重头的参数搜索。不能据此否定所有跨目标关系：辅助模型是微型MLP，尚未完整交叉拟合原冠军POR/PERM骨干。原64井OOF非默认数值预测均值与真值Spearman相关为POR0.9183、PERM0.9358、SW0.6427；这只说明辅助信号学到规律，不代表有独立增量。

## 未完成及计划
当前强POR/PERM模型的完整嵌套交叉拟合、更换协同输出范围、学习型跨井检索、输出序列模型与新井确认均未实施。不自动启动其他方向或原其他四折。原第五折仍是长期开发集，完整OOF>82目标未完成。

## 方法边界
辅助预测来源是按井OOF，原SW主模型仍在原64井训练，故全系统并非独立OOF。训练侧为单组辅助输出，原第五折为四组描述均值，存在方差差异；平均分位数不是混合分布分位数。shuffled仅作负对照，不可部署。前述所有限制保留在METHOD与FINAL。

## 维护与产物
FINAL.md、assembled_report.json、CURRENT_RECIPE.json、auxiliary_manifest.json、feature_manifest.json、timing.json及execution_notes.json已写入。辅助模型和来源描述作为可复用交叉拟合依赖保留，不覆盖旧模型/数据/已交付包。图谱使用curate_cross_target_v34.py增量维护并刷新build_atlas.py。所有训练已结束，无自动重启或排队。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

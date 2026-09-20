# external/pdda2021 · SPWLA PDDA 2021（Volve）迁移验证

上游仓库：<https://github.com/pddasig/Machine-Learning-Competition-2021>（数据 `data.zip`、`real_test_result.csv`，CC BY-NC-SA；本仓库不含数据）。
总结论文：Fu et al., 2024, *Petrophysics* 65(1) 108–127, doi:10.30632/PJV65N1-2024a6。

- `run_transfer_pdda2021.py`：数据适配（ft→m、μs/ft→μs/m、inch→cm、中子小数→%）、教程随机森林基线、留一井 CV（回归 / 分布模型两种解码 / 序列网络 / SW=1 门控）、Volve 可辨识性审计、盲井评估。
- `run_transfer_pdda2021_pass2.py`：盲井第二遍（回归 + 门控组合、增强特征、季军的物理范围规则、逐井分解）。

结果见 `docs/report_v3_optimization.md` §19 与 `results/transfer_pdda2021_*.json`。要点：解码跟指标走、池化胜最近邻井在 Volve 上复现；
序列网络判层优势在 4.2 万标签行下不复现；盲井 RMSE 0.106（冠军 0.0525），被 97.6% 为水层的 15/9-F-11T2 井拖垮。

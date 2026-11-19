# 公开 Volve 复核实验

## 实验目的

仅使用公开 PDDA 2021/Volve 训练井，复核 RF 基线、目标专用树回归、60-bin 分布均值/容差动作及联合 SW 门控。输出真实预测、逐井结果、论文表格和图件。

这不是新的未触碰盲测：公开测试井曾用于早期开发。Volve 为 VSH/PHIF/SW 与 RMSE，不包含主任务 PERM；容差指标仅为补充敏感性分析。

## 本次服务器执行

- 脚本：`reproduce_public_validation.py`
- 运行时间：约218秒
- 训练：公开训练井；318,967行，其中42,309行三标签完整
- 测试：公开测试井100、101、103、104，共11,275行
- 设备：NVIDIA H20
- Python 3.10.12；CatBoost 1.2.10；scikit-learn 1.7.2；NumPy 2.2.6；pandas 2.3.3

当前服务器复现命令：

```bash
cd <PROJECT_ROOT>
.venv/bin/python paper_workspace/public_volve/reproduce_public_validation.py
.venv/bin/python paper_workspace/public_volve/capture_environment.py
```

脚本依赖项目内的 `run_transfer_pdda2021.py` 和 `package/src/feature.py`、`package/src/utils.py`；这些依赖的哈希已记录。若对外发布，需要把项目根路径改为命令行参数并打包所需源码，不能把当前服务器硬编码路径宣称为已经完成的通用发布工具。

## 结果文件

| 文件 | 用途 |
|---|---|
| `metrics.json` | 总体与逐井RMSE、容差效用、数据划分与局限 |
| `metrics.csv` | 论文表的机器可读快照 |
| `predictions.npz` | 真实标签、井号、深度和各变体预测，均为公开数据 |
| `experiment_manifest.json` | 脚本、输入和输出的SHA256、时间、种子、限制 |
| `environment.json` | 实验后采集的环境、GPU和依赖源码哈希 |
| `fig_public_volve.pdf/png` | 论文公开结果图 |
| `public_table.tex` | 从metrics.json生成的论文表 |
| `make_public_table.py` | 表格生成和哈希核验脚本 |
| `run.log` | 服务器运行日志 |

## 主结果

- RF 五日志基线：RMSE 0.110819；
- 树回归：RMSE 0.107885；
- 树回归＋联合SW门控：RMSE 0.106187；
- 分布均值：RMSE 0.118471；
- 分布容差动作：RMSE 0.114797。

门控相对RF和未门控树回归的RMSE分别下降4.18%和1.57%，但Well 101上门控略差。PHIF容差效用从树回归的0.40916提高到分布容差动作的0.51583；这不能用来声称RMSE也同步改善。

## 数据与公开许可

来源：[PDDA 2021官方仓库](https://github.com/pddasig/Machine-Learning-Competition-2021)。公开数据许可与引用要求以官方说明为准。本文没有使用任何保密训练井或拟合模型生成上述公开结果。

当前为论文工作用可复核包，公开代码仓库和DOI尚未创建，正文已明确标注这一状态。

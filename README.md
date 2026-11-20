# Tolerance-Aware Reservoir Property Prediction from Wireline Logs

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](reproducibility/public_volve/)
[![Source of Truth: paper/current](https://img.shields.io/badge/Manuscript-paper%2Fcurrent-orange.svg)](paper/current/)

> **Tolerance-Aware Reservoir Property Prediction from Wireline Logs: Representation Complementarity, Decision Sensitivity, and Offline Transfer**  
> *Minjie Chen, Jiawei Xin, Chengfeng Wu*  
> Preprint submitted to *Geodata and AI*

---

## 概述 (Overview)

在地下储层表征中，利用电缆测井曲线定量预测储层物性参数——孔隙度（$\mathrm{POR}$）、渗透率（$\mathrm{PERM}$）与含水饱和度（$\mathrm{SW}$）——面临连续回归目标与离散默认态边界共存、以及实际工程容差评价准则与传统均方误差（MSE）损失函数系统性失配的严峻挑战。

本项目开源了论文的完整研究代码、公开基准复现流程、受控消融实验以及全套中英文手稿。我们提出了一种模块化预测体系，通过解耦**离散默认态门控分类**、**连续分位数目标分布估计**与**期望效用动作解码**，将预测概率分布直接与工业容差评价准则精准对齐。该方法在两个相互独立的离线油田基准（公开北海 Volve/PDDA 基准与包含 80 口井的大规模工业基准）上完成了系统评测与受控消融。

---

## 核心科学发现 (Key Findings)

1. **表征互补性与地质物理分工 (Functional Division of Labor)**：
   - **薄互层与界面分界（时序网络主导区）**：传统逐点（point-wise）决策树受制于测井仪器纵向分辨率极限与围岩肩效应（shoulder-bed effects），在薄互层（$<0.5\,\mathrm{m}$）、岩性突变过渡带及储层流体界面处容易产生过渡态误判；而深层时序网络依托大跨度纵向感受野感知沉积旋回，在突变边界门控分割上显著优于树模型（$\mathrm{SW}$ 状态分类准确率从 0.867 提升至 0.923）。
   - **均质厚层物性回归（决策树主导区）**：在大套均质储层段内部，局部多参数非线性交会占主导且无边界干扰，梯度提升决策树（GBDT）在连续数值回归上精度与抗噪性均超越深度模型。
2. **严格受控消融与参数稳定性 (Controlled Ablations & Sensitivity)**：
   - 在基学习器完全冻结、共享交叉验证折的受控条件下，剥离单一模块的因果贡献。
   - 系统遍历连续融合权重、决策阈值与渗透率分位数分箱分辨率（$K \in \{30, 60, 120\}$），证实架构在宽泛参数带内具备高度收敛与抗扰稳定性。
3. **离线跨工区迁移边界 (Transferability Boundaries)**：
   - 在公开北海 Volve 基准（9 口训练井）上，目标专用树模型配合 $\mathrm{SW}$ 联合门控取得了领先的合并 RMSE（0.10619）；
   - 高容量 18 阶段双主干时序流水线在稀疏井网环境下出现明显的泛化衰退（RMSE 为 0.14695），明确界定了深度时序模型在小样本工区应用的数据规模依赖边界。

---

## 目录结构 (Repository Structure)

```
welllog-reservoir-hybrid/
├── paper/                          # 论文手稿与补充材料
│   ├── current/                    # 英文投稿稿件（Source of Truth，24 页主文 + 7 页补充材料）
│   │   ├── main.pdf / main.tex     # 英文主论文 PDF 与 LaTeX 源码
│   │   ├── supplement.pdf / .tex   # 英文补充材料 PDF 与源码
│   │   ├── figures/                # 论文全部矢量图件与绘图源数据
│   │   └── manifest.json           # 稿件文件 SHA-256 完整性清单
│   ├── chinese/                    # 中文完整对照版本（20 页主文 + 7 页补充材料）
│   │   ├── main_zh.pdf / .tex      # 中文主论文 PDF 与源码
│   │   └── supplement_zh.pdf / .tex# 中文补充材料 PDF 与源码
│   └── README.md                   # 论文目录说明
├── reproducibility/                # 公开 Volve/PDDA 基准复现管线
│   └── public_volve/               # 一键复现脚本、评测指标、运行环境与生成图件
├── experiments/                    # 实验源码与消融分析
│   ├── reviewer_response/          # 留一模块消融 (LOMO)、门控敏感性、分箱分辨率实验
│   └── public_v4/                  # 公开 Volve 上的 v4 双主干从头训练流水线
├── tools/                          # 图件渲染、发布审计与去敏感化工具
├── docs/                           # 技术备忘、审稿回应与历史演进记录
│   └── history/                    # 早期历史草稿与阶段性工作记录
├── CITATION.cff                    # Citation File Format 引用元数据
├── RELEASE_AUDIT.md                # 发布审计与完整性报告
└── RELEASE_MANIFEST.json           # 全库文件发布哈希清单
```

---

## 快速开始：公开 Volve 基准复现 (Quickstart)

公开北海 Volve/PDDA 基准可在常规 CPU/GPU 环境下快速复现（无需任何商业授权数据）：

### 1. 环境准备

```bash
git clone https://github.com/whiteicey/welllog-reservoir-hybrid.git
cd welllog-reservoir-hybrid
pip install -r requirements.txt
```

### 2. 获取公开数据

从 [SPWLA PDDA SIG 官方仓库](https://github.com/pddasig/Machine-Learning-Competition-2021) 下载公开竞赛数据，将数据放置于 `external/pdda2021/` 目录：
- `train.csv`（9 口训练井）
- `test.csv`（4 口测试井盲样）
- `test_labels.csv`（已公开发布的测试井标签）

### 3. 执行复现与敏感性分析

```bash
# 1. 运行五模型公开基准复验（生成测试集分井与合并 RMSE）
python reproducibility/public_volve/reproduce_public_validation.py

# 2. 运行 SW 联合门控阈值敏感性遍历
python reproducibility/public_volve/public_gate_sensitivity.py

# 3. 重新绘制公开基准综合对比图（Figure 1）
python tools/make_public_figure.py
```

执行产物与图件将自动保存至 `reproducibility/public_volve/output/`。

---

## 内部基准复核与数据政策 (Data Policy & Governance)

本研究严格遵循学术规范与数据合规要求：
- **公开数据**：Volve/PDDA 数据遵循 Equinor Volve 开源协议，测试代码与真值完全公开；
- **工业数据**：内部 80 口井工业数据集受企业保密协议与数据合规限制，不包含在公开仓库中。为保证实验透明度与科学可复核性，`experiments/reviewer_response/` 提供了全部受控消融与超参数遍历脚本、冻结配置参数、环境记录以及逐折输出的聚合 JSON 结果与哈希值。拥有授权的研究人员可通过设置 `WELLLOG_ROOT` 环境变量在本地数据环境上直接重跑复验。

---

## 引用 (Citation)

如果您在研究中参考了本项目的算法设计、实验数据或论文成果，请按如下格式引用：

```bibtex
@article{chen2026tolerance,
  title={Tolerance-Aware Reservoir Property Prediction from Wireline Logs: Representation Complementarity, Decision Sensitivity, and Offline Transfer},
  author={Chen, Minjie and Xin, Jiawei and Wu, Chengfeng},
  journal={Geodata and AI},
  year={2026},
  note={Minjie Chen and Jiawei Xin contributed equally to this work.}
}
```

---

## 许可证 (License)

本项目开源代码遵循 [Apache 2.0 License](LICENSE)。论文手稿与图件版权归作者所有。

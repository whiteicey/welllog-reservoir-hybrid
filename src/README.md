# src · v3 实验代码说明

所有脚本在服务器项目根目录运行（`data/` 为赛事数据软链接，`cache/` 为特征缓存，`outputs/` 为实验产物）。
共同约定：`harness.py` 提供数据/特征缓存、按井 5 折切分（固定）、官方指标、三角核效用矩阵、Bayes 动作、配对井 bootstrap。
每个实验写 `outputs/<name>/{oof.npz,test.npz,report.json}`；`test.npz` 保存每折模型对 10 口测试井的输出，供融合时平均。

## 运行顺序

| 步骤 | 脚本 | 说明 | 耗时（H20） |
| --- | --- | --- | --- |
| 0 | `harness.py` | 首次调用自动构建 `cache/`（113/237 维特征、折号、原始曲线） | 5 min |
| 1 | `cv_v2.py` | 复现 v2 的 12 个模型 5 折 OOF；所有比较的基准 | 45 min |
| 1b | `cv_v1.py` | 复现 v1（固定 500 轮估计器）5 折 OOF | 15 min |
| 2 | `diag_wells.py`、`diag_runs.py`、`diag_gate.py`、`diag_hf.py`、`diag_wellscale.py`、`diag_typewell.py` | 可辨识性审计、段结构、门控解剖、高频可解释性、每井缩放、类型井选井 | CPU 10～40 min 各 |
| 3 | `exp_decisions.py`、`exp_calib_joint.py` | 解码规则、门控平滑、嵌套阈值、校准 + 联合解码（无需重训） | CPU 10 min |
| 4 | `exp_variant.py --name X --target T --model bins/reg_mae/reg_rmse/gate --features base|enh[+extra][+stack][+aln]` | CatBoost 变体（分布模型 / 回归 / 门控） | 15～60 min |
| 5 | `exp_states.py --features base+extra[+aln]` | 4 状态联合模型 | 15 min |
| 6 | `nn_seq.py --name X [--use-base 1] [--pseudo-oof ...]` | 整井序列网络 5 折（门控头 + 分箱头） | 1.3 h / 3 种子 |
| 7 | `stack_states.py` | 二级堆叠状态模型（343 维） | 15 min |
| 8 | `assemble.py --config assemble_cfg_X.json --out outputs/assembly_X` | 嵌套融合：分布混合 + 门控混合 + 阈值；输出 `decision.json` | 20～60 min CPU |
| 9 | `train_final.py --config final_cfg_v3.json --out-dir outputs/final_v3` | 全量训练（CatBoost + 序列网络），写 `model_meta.json` | 2.5 h |
| 10 | `build_submission_v3.py` | 按官方目录模板打包，独立解压重推理并比对 SHA | 5 min |
| 11 | `make_figures.py`、`make_supplement.py`、`collect_results.py` | 论文图、补充材料、结果汇总 | 分钟级 |

其他：`extra_features.py`（181 维附加特征）、`dtw_align.py`（井间相似度 + DTW 对齐特征）、`make_ablations.py`（消融提交文件）、
`queue_*.sh`（实验队列）、`sync.sh`（本地→服务器同步）、`run_transfer_pdda2021*.py`（Volve 迁移验证，见 `external/`）。

## 融合配置文件

`assemble_cfg_{A..I}.json`：每个目标的分布来源（`dist_sources`）与门控来源（`gate_sources`）。D 为提交配置；
E 为去网络消融；F/G/H/H0/I 为饱和检验。`final_cfg_v3.json` 把 D 的决策块与全量训练超参数绑定。

## 依赖

`../requirements.txt`（服务器 venv）：catboost 1.2.10、torch 2.14 (cu126)、lightgbm 4.7.0、scikit-learn 1.7.2、numpy 2.2.6、pandas 2.3.3、scipy 1.15、dtaidistance、matplotlib。
`ref/solution_v2/src/` 是 v2 的数据读取与 113/237 维特征代码，`harness.py` 直接导入它以保证特征与 v2 一致。

# 纯深度学习训练入口

此目录为现有冻结实验提供可移植的路径和进程检查封装。训练仍执行 `experiments/pure_dl/` 内的原文件，不改写模型、输入特征、损失、解码、评分、井分组、种子或原源码哈希。

## 使用

从仓库根目录执行，使用已安装 CUDA 版 PyTorch 的 `normal` 环境：

```powershell
conda activate normal
python tools/deep_learning/train.py --help
python tools/deep_learning/train.py --method posterior-router --data_dir D:/match_predict/data --out outputs/reproduction_posterior_router_01 --fold-count 1
```

Linux 使用相同参数，例如将 `--data_dir` 改为 `/datasets/welllog`。也支持 `python -m tools.deep_learning.train`。`--data_dir` 指向包含 `train/*.txt` 的目录，保留原官方文件名、两行表头、井集合与文件内容；不传 `train` 子目录本身。`--out` 必须不存在或为空，不复用已有结果目录。运行帮助页只需 Python 标准库，不读取数据、不导入 PyTorch、不访问 GPU。

训练依赖 Python 3.10+、CUDA 版 PyTorch（原要求 `torch>=2.4`）、NumPy、pandas、scikit-learn 和 psutil。实际训练沿用原脚本的 CUDA BF16 运算，需要兼容的 GPU。scikit-learn 会安装 SciPy；原仓库完整 requirements 另包含 CatBoost。虽然 `train_invariant` 导入的 `train.py` 会导入 `src.model`，其中 CatBoost 仅在 `load_cb` 内惰性导入，本入口不会调用树模型，无需因此单独安装 CatBoost。环境安装和驱动配置不由此入口执行。

## 方法与冻结预算

| `--method` | 原训练文件（位于 `experiments/pure_dl/`） | 最大轮数 / patience | 原文件的筛选门槛 |
|---|---|---:|---:|
| `calibrated75` | `train_paired_calibrated_view_sequence.py` | 75 / 18 | 严格 >80.2 |
| `calibrated80` | `train_paired_calibrated_view80_sequence.py` | 80 / 18 | 严格 >80.2 |
| `router` | `train_paired_detached_router80_sequence.py` | 80 / 18 | 严格 >80.2 |
| `local-router` | `train_paired_local_context_router80_sequence.py` | 80 / 18 | 严格 >80.4 |
| `posterior-router` | `train_paired_posterior_router80_sequence.py` | 80 / 18 | 严格 >80.4 |

统一传入原合法参数 `--learning-rate 0.0006 --schedule cosine`。其余设置保留原脚本：minimum epochs 10、每轮 64 步、每步 64 个窗口、seed 20261006，固定第五折使用 seed+4。原脚本直接恢复本折验证官方 TOTAL 最佳 checkpoint，不按最佳轮数另行重训。75 与 80 轮的 cosine 周期不同，不能声称前期训练轨迹相同。`calibrated75` 是明确命名的历史复现配置，新增探索应使用 80 轮配置。

## 单折与完整五折的边界

`--fold-count 1` 只运行 **原第五折（零基索引 4）**，不是第一折；原 `GroupKFold(5)` 的 80 口井、723,556 行有效标签和行序约束由训练脚本保留。原输出 `oof.npz` 仅第五折预测为有限数值，其余四折为 NaN；`screening_report.json` 标记 `screening_fold=5`。这不是完整五折 OOF 成绩。

此入口明确拒绝 `--fold-count 5`，不会自动晋级、生成 `screening_verification.json` 或伪造完整复现证明。完整续跑仍需原生流程中与数据、配置、源码、缓存对应的独立门槛证明；可移植核验入口尚未整合。当前新候选晋级门槛为严格 **>80.4**，旧脚本的 `screening_passed` 可能只表示历史 >80.2；保留这个旧字段不等于允许按旧门槛续跑。历史 calibrated75 的完整运行与历史授权不被重新解释为 >80.4 晋级。

## 路径、GPU 检查与证据

封装在内存中临时替换两处函数，结束或异常时恢复：

1. `train_invariant.load_split`：调用同一个原始读取函数，仅把目录指向 `--data_dir`，只允许 `train` 分区。
2. `gpu_job_guard_v2.check_slot`：改用本目录的跨平台 `psutil` 检查，识别 Windows/Linux Python 的原生 `train_*.py`、`benchmark_*.py`、`finetune_*.py`、相关 `gen_*` 脚本以及本训练入口。按 PID 去重，最多两个作业，包含当前进程；当前 CUDA 设备空闲显存至少 6 GiB。

原生 `run_*.py` 等待父进程不重复计数，其实际训练子进程会被计数。本入口通过 `runpy` 在同一进程执行训练，只有一个 PID。进程检查是启动时的**非原子快照**，不是任务预约锁；不要同时抢占最后一个名额。任意 `python -c`、第三方训练入口及不可识别进程不在完整检测承诺内；已识别但无法读取命令行的 Python 进程会使检查拒绝启动。

`launcher_adjustments.json` 单独记录方法、数据路径、平台、预算、原/当前门槛、两项覆盖、原训练文件与两种 guard/入口的 SHA256、运行状态。原训练 `config.json` 的依赖哈希不被篡改，但它本身无法描述内存中的 guard/path 替换，因此复现证据应连同 sidecar 保存。`trainer_returned` 只表示原脚本返回，不表示已独立核验成绩，也不表示完整五折成功。

跨平台入口不等于跨驱动/硬件的逐位一致性证明。原数据清洗、输入派生、训练折统计以及单折结果的有效性仍遵守原实验约束。

## 本次整理记录

- **重点与假设**：把硬编码的数据根目录和 Windows 进程名检查隔离到入口，在冻结源码上提供明确的单折使用方式。
- **已经完成与验证**：实现五种映射、第五折限制、路径重定向、GPU 作业快照与独立 sidecar；仅进行语法及 `--help` 包装检查。未启动训练、未执行 GPU 基准，也未重新核验历史成绩。
- **收益**：消除入口对本机数据绝对位置和 `python.exe` 名称的依赖，保留原脚本证据。训练速度、分数变化及 Linux 实际训练可运行性尚未实测，不宣称数值收益。
- **未继续的范围**：不复制/重写算法，不调整超参数，不绕过五折晋级证明；这些改动会扩大复现差异，超出入口整理范围。项目进展及探索索引由本批仓库整理任务统一维护。

2026-09-26 包装检查已通过：在 `normal` 环境用 `compile(..., 'exec')` 检查两份 Python 文件（不生成字节码），并运行 `python tools/deep_learning/train.py --help`，均退出 0。未执行真实数据训练、GPU 检查或 Linux 运行验证。

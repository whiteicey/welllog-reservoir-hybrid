# 提交产物

| 目录 | 内容 | A 榜 |
| --- | --- | ---: |
| `v1/` | 首轮提交：`result.zip`、`model_meta.json`（113 维、3 折 OOF 参数） | 81.0052 |
| `v2/` | 目标专用方案：`result.zip`、`model_meta.json`、`checksums.sha256` | 82.2718 |
| `v3/` | 原 v3：`result.zip`（SHA `6694a7a6…`）、`model_meta.json`、`checksums.json`、`ablations/` | 83.2579 |
| [v3.5/](v3.5/README.md) | 原结果 ZIP、原代码 ZIP、版本与来源 manifest；未附权重，未独立推理 | **83.2960**（用户反馈） |

v3.5 结果已在本机校验 10 井、95,948 行；内部 `version=3.0` 原样保留，不代表它与原 v3 使用相同模型。新榜分未独立平台核验，详见 [v3.5 归档](v3.5/README.md)。

队友 `deep_learning` 的 **83.6915** 对应校准双骨干；当前 Git 不含该交付的匹配 result.zip/五折权重，本次只保存[分支证据快照](../results/comparison/README.md)，不把其他分支候选冒充该结果。

`v3/submission_code.zip`（404 MB，官方目录模板，含 19 个权重文件）不在 git 内，见 GitHub Release **v3.0** 附件；
SHA256 `47f4ada44b29c99f6b4420ade01ce41bd9bcb969bfb13a26763883ff28857a5d`。从该包独立解压、只用 `test/` 重推理得到的 `result.json`
与 `v3/result.zip` 内文件逐字节一致（SHA `afcaad7612fac1cea0413d2a594532c3103b9af06d5f960d77c26c479f9e8358`）。

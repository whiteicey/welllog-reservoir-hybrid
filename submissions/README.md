# 提交产物

| 目录 | 内容 | A 榜 |
| --- | --- | ---: |
| `v1/` | 首轮提交：`result.zip`、`model_meta.json`（113 维、3 折 OOF 参数） | 81.0052 |
| `v2/` | 目标专用方案：`result.zip`、`model_meta.json`、`checksums.sha256` | 82.2718 |
| `v3/` | 最终方案：`result.zip`（A 榜提交文件，SHA `6694a7a6…`）、`model_meta.json`（决策配置）、`checksums.json`、`ablations/`（槽 2～4 消融提交文件） | **83.2579** |

`v3/submission_code.zip`（404 MB，官方目录模板，含 19 个权重文件）不在 git 内，见 GitHub Release **v3.0** 附件；
SHA256 `47f4ada44b29c99f6b4420ade01ce41bd9bcb969bfb13a26763883ff28857a5d`。从该包独立解压、只用 `test/` 重推理得到的 `result.json`
与 `v3/result.zip` 内文件逐字节一致（SHA `afcaad7612fac1cea0413d2a594532c3103b9af06d5f960d77c26c479f9e8358`）。

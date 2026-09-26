# 纯 DL 独立推理源码

本目录是已交付 `81.713233` 完整五折版本的推理源码。**Git 不包含权重**，不需要训练标签或联网服务。

## 环境与运行

已实际验证 Python 3.10.11、PyTorch 2.7.0+cu128、NumPy 1.26.4、pandas 2.2.3（Windows GPU）。依赖见 `requirements.txt`；GPU 机器需安装与驱动兼容的 PyTorch wheel。其他平台与本版本 CPU 路径未实测。

把五个模型文件 `fold_1.pt` … `fold_5.pt` 放到 `models/`，与已附的 `feature_contract.json` 放在一起。权重 SHA256、最佳轮次和分数见 [model_meta.json](models/model_meta.json)。Git 克隆本身不能提供这些权重；本地保存在独立交付目录 `submission_calibrated_pair/models/`。

```bash
python -m pip install -r package_dl/requirements.txt
python package_dl/predict.py --data_dir /path/to/data --output /path/to/result.json --device cuda --batch_windows 32
```

也可指定 `--model_dir /path/to/models`。`--data_dir` 接受包含 `test/` 的根目录或测试 TXT 文件夹。输入原始井格式两行表头，缺失值标记由解析器处理。原始深度行顺序保持不变。无需修改源码或手工注入数据。

## 模型与输出

每折有两个骨干：206 维完整特征分支与经过逐点校正的 68 维分支。每折完整后验先等权融合，再对五折期望官方效用等权平均并解码，不能平均五折已经解码的数值。九条实际曲线和算法细节见 [METHOD.md](../docs/deep_learning/METHOD.md)。

`result.json` 严格使用以下字段，不含额外版本或模型字段：

```json
{"resultData":[{"logId":"well_id","predictions":[{"DEPTH":505.0,"POR":0.1,"PERM":12.35,"SW":99.9}]}]}
```

同时输出 `result.zip`（仅有根目录 `result.json`）及运行统计 `result.inference.json`。SW 保持原数据的数值尺度，不额外除以 100。

原五折交付以冻结的 75 轮、patience=18 训练，并保留各折验证最优 checkpoint。它不同于后续的 80 轮路由探索。完整 OOF 为 81.713233，测试无标签、成绩未知。两次完整 GPU 推理的结果字节一致；验证摘要见 [results/deep_learning](../results/deep_learning/README.md)。

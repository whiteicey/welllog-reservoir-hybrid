# v41 PERM分布族对照完成

## 探索重点与条件

冻结当前独立PERM预测器，两个同初始化11824参数小头修正16分量权重、位置与尺度。只改变Gaussian或Student-t3密度及对应CDF；t3 scale=SD/sqrt(3)，初始分量均值/标准差相同但分布不相同。默认概率固定，原log10标签、Q50、原9曲线/206主特征、数据划分、支持网格和官方动作/评分保持。
80轮上限/patience18/min10/batch64/64步，两头独立优化和早停。相同训练窗口，AdamW3e-4，原预测器权重全部冻结。

## 已探索与验证

|候选|初始PERM|训练后最高PERM|训练后减初始|训练后减原模型|
|---|---:|---:|---:|---:|
|gaussian|89.396724049366|89.367194812859|-0.029529236507|-0.029529236507|
|student3|89.376497588646|89.372730778423|-0.003766810223|-0.023993270943|

两组均第18轮早停，训练后最高均在第16轮；候选自身最佳均为第0轮，最终都回退原PERM。Gaussian初始全144665点逐点相同；两候选恢复、冻结权重和源哈希核对通过。输入/输出层均有实际梯度和参数变化。

## 取得的收益与解释

固定POR/PERM/SW实际三列评分TOTAL=80.336513436816，相对父配方+0。训练后最佳t3-Gaussian=+0.005535965564 PERM，但t3初始已低于父模型，继续训练也低于自身初始。因此该局部差异不构成采用t3的正收益；单种子/重复开发折不作稳定性结论。

## 不继续的点与证据

当前有界小头两种分布均未改善原PERM89.396724；不继续自由度、混合系数或解码网格扫描。结论限当前冻结预测器与小头/NLL配置，不能推广为所有重尾分布无效。

## 耗时

{
  "training_and_evaluation_seconds": 95.24599619999935,
  "peak_allocated_MiB": 716.8544921875,
  "launch_to_completion_seconds": 100.02865,
  "processes": 1,
  "candidates": 2,
  "scope": "Two independent heads, sequential head updates with shared frozen forward in one GPU process. Launch wall time includes load/preparation/warmup/final restore; loop time excludes initial loading. Existing data cache reused. Authoring/finalization excluded.",
  "finalization_seconds": 0.5078307999938261
}

## 目标与未做

第五折严格>80.5和完整原五折OOF>82仍未完成。原其他四折、旧队列均未启动，旧交付保留。当前配方沿用v31。后续若继续，应先提出不同机制的有界实验，避免再次扩展本次分布族小头扫参。
结果见assembled_report.json、CURRENT_RECIPE.json；训练源/配置/逐轮记录与紧凑预测保留，负候选权重按删除清单清理。

## v41负权重清理
已完成恢复和实际三列评分，删除本实验负候选checkpoint共5个、7224100字节；源/配置/训练日志/紧凑预测和原冠军依赖保留。清单见perm_family_v41_20260929/deletion_manifest.json。无活动训练或自动队列。


# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v71 SW深度关系注意力

## 探索重点和预定义对照
本轮先核对旧特征：原206已有差分/导数，directional_depth_features.py及已关闭试验覆盖1m/4m物理深度方向统计，故不重复该输入特征。改在当前v56 SW注意力内部学习深度关系，冻结全部原模型，仅4层×4头×8系数=128新参数。
symmetric仅用物理深度差绝对值，directional保留上下方向。均先以10m尺度tanh压缩，8个固定归一化RBF基（各自[0,1]/[-1,1]等距，带宽为一个基间距），偏置是0.5*tanh系数的凸组合，绝对值≤.5。只差关系几何；同参数值/数量及零偏置完整起点，不声称函数空间相同。
原始深度仅用于窗口中心化后的token均值与两两差，平移不改变偏置；不恢复绝对深度、DEVI/AZIM/BIT/CASE输入或井ID。无跨井token，原valid key掩码保留。原注意力加上带偏置与零偏置两次相同浮点mask实现之差，零偏置时差为0，避免改变原布尔mask实现的初始数值函数。此实现每块三次attention，真实开销需记录，不能宣传为免费。

## 数据训练边界与已做
原9曲线/206输入/训练64井预处理/Q50/第五16井144665点/seed20261010/1024窗口中心128/原Gaussian SW损失/官方解码评分保持。原v56全部参数冻结eval，但新偏置经原冻结网络的输入梯度保持；仅128系数AdamW3e-3/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps。零系数不重置原预测，回退不是学习收益。POR/PERM不改。
已实现，未训练/无成绩。直接运行真实四组系数梯度、原权重不变、初始与最佳144665点及三列评分验证，源哈希固定；正候选保留前磁盘重建，无独立数值预检/审查。原子完整last_state包含优化器/调度/RNG/停止，仅显式恢复。

## 收益、待做与不继续
两项训练/配对及绝对差、清理/PROJECTS/EXPLORATIONS/图谱待完成。若directional无最终配对收益，不扫距离尺度/RBF数量；若两者都无当前收益，不继续偏置幅度扫描。仅限当前粗token物理关系，不泛化全部位置编码。
最多2GPU，按实际显存决定第二项启动，不自动队列；记录单项与实际总墙钟、并发和显存，包含加载/预热/三次attention训练/评估，代码和收尾另计。负检查点先清单后删，保留原正模型/交付。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成，无其他折/旧任务/交付修改。目标后独立确认、新目录交付。


## v71实际后端失败与修复
原symmetric首次反向失败：RuntimeError: LSE is not correctly aligned (strideH)，已完成初始144665点一致但无训练轮次。原源码、配置、failure/status保留且不改；math新版本改新增两支为显式FP32矩阵注意力，原注意力保持，重新随机状态同零初始化启动，不声称旧轨迹续跑。


## v71最终结果
两项完成，TOTAL=80.354325927816；directional-symmetri

# v71 SW深度关系注意力

## 探索重点和预定义对照
本轮先核对旧特征：原206已有差分/导数，directional_depth_features.py及已关闭试验覆盖1m/4m物理深度方向统计，故不重复该输入特征。改在当前v56 SW注意力内部学习深度关系，冻结全部原模型，仅4层×4头×8系数=128新参数。
symmetric仅用物理深度差绝对值，directional保留上下方向。均先以10m尺度tanh压缩，8个固定归一化RBF基（各自[0,1]/[-1,1]等距，带宽为一个基间距），偏置是0.5*tanh系数的凸组合，绝对值≤.5。只差关系几何；同参数值/数量及零偏置完整起点，不声称函数空间相同。
原始深度仅用于窗口中心化后的token均值与两两差，平移不改变偏置；不恢复绝对深度、DEVI/AZIM/BIT/CASE输入或井ID。无跨井token，原valid key掩码保留。原注意力加上带偏置与零偏置两次相同浮点mask实现之差，零偏置时差为0，避免改变原布尔mask实现的初始数值函数。此实现每块三次attention，真实开销需记录，不能宣传为免费。

## 数据训练边界与已做
原9曲线/206输入/训练64井预处理/Q50/第五16井144665点/seed20261010/1024窗口中心128/原Gaussian SW损失/官方解码评分保持。原v56全部参数冻结eval，但新偏置经原冻结网络的输入梯度保持；仅128系数AdamW3e-3/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps。零系数不重置原预测，回退不是学习收益。POR/PERM不改。
已实现，未训练/无成绩。直接运行真实四组系数梯度、原权重不变、初始与最佳144665点及三列评分验证，源哈希固定；正候选保留前磁盘重建，无独立数值预检/审查。原子完整last_state包含优化器/调度/RNG/停止，仅显式恢复。

## 收益、待做与不继续
两项训练/配对及绝对差、清理/PROJECTS/EXPLORATIONS/图谱待完成。若directional无最终配对收益，不扫距离尺度/RBF数量；若两者都无当前收益，不继续偏置幅度扫描。仅限当前粗token物理关系，不泛化全部位置编码。
最多2GPU，按实际显存决定第二项启动，不自动队列；记录单项与实际总墙钟、并发和显存，包含加载/预热/三次attention训练/评估，代码和收尾另计。负检查点先清单后删，保留原正模型/交付。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成，无其他折/旧任务/交付修改。目标后独立确认、新目录交付。


## v71实际后端失败与修复
原symmetric首次反向失败：RuntimeError: LSE is not correctly aligned (strideH)，已完成初始144665点一致但无训练轮次。原源码、配置、failure/status保留且不改；math新版本改新增两支为显式FP32矩阵注意力，原注意力保持，重新随机状态同零初始化启动，不声称旧轨迹续跑。


## v71最终结果
两项完成，TOTAL=80.354325927816；directional-symmetric SW=+0.000000000000，训练后差=-0.018606844709；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_depth_attention_v71_20260929/FINAL.md。


## FINAL.md

# v71 SW深度关系注意力结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含起点SW|最佳/停止轮|
|---|---:|---:|---|
|symmetric|73.711458293211|73.713975628494|0/18|
|directional|73.692851448502|73.713975628494|0/18|

## 探索重点与验证
原1m/4m方向差特征已被历史尝试覆盖，未重复训练该特征。本轮冻结v56全SW，仅4层×4头×8个零初始系数。物理token深度两两差/10m/tanh，symmetric取绝对值、directional保留正负，各自8个归一化RBF基；系数0.5*tanh凸组合令偏置绝对值≤0.5。相同128参数和完整起点，不声称两种函数空间相同。原数据/Q50/损失/1024中心128/解码和评分保持，新系数3e-3。
原始版本首次反向出现CUDA LSE strideH对齐错误、无完成训练轮；原失败来源/config/evidence冻结。math版保留原布尔mask注意力，仅新增偏置与零偏置差分用显式FP32 QK/softmax/V；零系数差为0，各自初始144665点与父模型逐位一致。该结构每块三次attention，成本实报。
四层实际系数梯度有效，全部原参数无梯度且状态哈希不变；同初始状态/骨干/头/预测、来源哈希、最佳144665恢复与真实三列评分核验；新采用者原始输入磁盘重建=False。

## 收益与不继续
directional-symmetric含起点SW差+0.000000000000；训练后最佳差-0.018606844709；采用=None。
起点回退不是学习收益。若方向关系无最终配对收益，不扫距离尺度/RBF数量；两组均无当前收益则不扫偏置幅度。仅限粗token物理关系，不泛化全部位置编码；重复开发折仍无独立确认。

## 耗时、失败、清理与未完成
{
  "candidates": [
    {
      "mode": "symmetric",
      "start": "2026-09-29T23:23:05.377250+08:00",
      "end": "2026-09-29T23:24:46.474416+08:00",
      "seconds": 96.93524409999372,
      "peak_allocated_MiB": 922.54052734375
    },
    {
      "mode": "directional",
      "start": "2026-09-29T23:24:25.110008+08:00",
      "end": "2026-09-29T23:26:11.947718+08:00",
      "seconds": 101.99892729999556,
      "peak_allocated_MiB": 922.54052734375
    }
  ],
  "first_launch_to_last_completion_seconds": 186.570468,
  "peak

# v71 SW深度关系注意力结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含起点SW|最佳/停止轮|
|---|---:|---:|---|
|symmetric|73.711458293211|73.713975628494|0/18|
|directional|73.692851448502|73.713975628494|0/18|

## 探索重点与验证
原1m/4m方向差特征已被历史尝试覆盖，未重复训练该特征。本轮冻结v56全SW，仅4层×4头×8个零初始系数。物理token深度两两差/10m/tanh，symmetric取绝对值、directional保留正负，各自8个归一化RBF基；系数0.5*tanh凸组合令偏置绝对值≤0.5。相同128参数和完整起点，不声称两种函数空间相同。原数据/Q50/损失/1024中心128/解码和评分保持，新系数3e-3。
原始版本首次反向出现CUDA LSE strideH对齐错误、无完成训练轮；原失败来源/config/evidence冻结。math版保留原布尔mask注意力，仅新增偏置与零偏置差分用显式FP32 QK/softmax/V；零系数差为0，各自初始144665点与父模型逐位一致。该结构每块三次attention，成本实报。
四层实际系数梯度有效，全部原参数无梯度且状态哈希不变；同初始状态/骨干/头/预测、来源哈希、最佳144665恢复与真实三列评分核验；新采用者原始输入磁盘重建=False。

## 收益与不继续
directional-symmetric含起点SW差+0.000000000000；训练后最佳差-0.018606844709；采用=None。
起点回退不是学习收益。若方向关系无最终配对收益，不扫距离尺度/RBF数量；两组均无当前收益则不扫偏置幅度。仅限粗token物理关系，不泛化全部位置编码；重复开发折仍无独立确认。

## 耗时、失败、清理与未完成
{
  "candidates": [
    {
      "mode": "symmetric",
      "start": "2026-09-29T23:23:05.377250+08:00",
      "end": "2026-09-29T23:24:46.474416+08:00",
      "seconds": 96.93524409999372,
      "peak_allocated_MiB": 922.54052734375
    },
    {
      "mode": "directional",
      "start": "2026-09-29T23:24:25.110008+08:00",
      "end": "2026-09-29T23:26:11.947718+08:00",
      "seconds": 101.99892729999556,
      "peak_allocated_MiB": 922.54052734375
    }
  ],
  "first_launch_to_last_completion_seconds": 186.570468,
  "peak_process_concurrency": 2,
  "finalization_seconds": 0.3449172999971779,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate",
  "failed_attempt": {
    "failed_directory": "outputs\\pure_dl_sw_depth_attention_v71_SW_symmetric_fold5",
    "original_error": "RuntimeError('LSE is not correctly aligned (strideH)')",
    "completed_epochs": 0,
    "action": "fresh math implementation version, no state resume; identical zero initial coefficients",
    "source_frozen": true
  },
  "first_failed_launch_to_last_completion_seconds": 274.544848
}
删除本轮负或已被替代失败检查点6个，共36716794字节，清单先写后删；所有源码/配置/失败/日志/预测保留。
math两项与含失败/修复间隔总墙钟分别记录；单项含加载/预热/三次attention训练/评估，代码/收尾另计。第五严格>80.5和完整原五折OOF>82未完成；无其他折/旧队列/交付修改。目标后独立确认和新目录交付。



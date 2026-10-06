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

# v59 完整RQS密度读出结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含起点SW|最佳/停止轮|
|---|---:|---:|---|
|affine|73.640665034391|73.708218105973|0/18|
|silu|73.678876918248|73.708218105973|0/18|

## 探索重点与完成验证
同851237总参数/14852密度参数与完整初始函数；冻结v58骨干/默认头，仅完整密度输出更新。仿射过参数化与SiLU表达能力不同。固定canonical训练上下文是共同变化，不能把历史绝对差全部归因于激活。
训练578891点、验证144665点分包，Q50与原RQS公式/官方评分保持；初始源预测、参数与头哈希相同，冻结状态和真实梯度已核验。保留候选均从原始输入重载完整模型逐点验证。
准备首轮因训练专用labeled标记用于验证而触发零验证覆盖断言，未训练；改用原评估器raw_to_valid映射后重建。失败证据cache_failure_01.json保留。

## 收益与不继续的点
SiLU-affine SW +0.000000000000；训练后最高差+0.038211883857；采用None，保留[]。
初始回退不算训练收益。若非线性无配对收益，关闭当前骨干的隐藏宽度/激活扫描；结论不泛化其他输出头。正局部收益仍是重复开发折证据，未经独立井确认。

## 耗时、清理和未做
{
  "candidates": [
    {
      "mode": "affine",
      "start": "2026-09-29T21:22:54.685217+08:00",
      "end": "2026-09-29T21:23:32.843106+08:00",
      "seconds": 33.707767600004445,
      "peak_allocated_MiB": 499.81982421875
    },
    {
      "mode": "silu",
      "start": "2026-09-29T21:22:59.543433+08:00",
      "end": "2026-09-29T21:23:36.051648+08:00",
      "seconds": 31.85266689999844,
      "peak_allocated_MiB": 499.81982421875
    }
  ],
  "cache_preparation_seconds": 6.22869059999357,
  "cache_preparation_peak_MiB": 506.6884765625,
  "pair_launch_to_completion_seconds": 41.366431,
  "first_preparation_including_failed_attempt_to_completion_seconds": 236.772189,
  "successful_preparation_to_completion_seconds": 188.002394,
  "gap_after_cache_to_first_launch_seconds": 139.262416,
  "finalization_and_raw_replay_seconds": 0.3018891999963671,
  "scope": "training includes load/warmup/evaluation; cache shared once; wall times include launch/authoring gaps; failed preparation separately preserved"
}
删除负权重与可再生缓存5项，共428683249字节。保留模型不依赖训练缓存推理，训练恢复需先重建同缓存。
目标仍是第五折严格>80.5及完整原五折OOF>82；本批无其他折、旧任务恢复或交付覆盖。未做独立新井确认。目标完成后可新目录交付。

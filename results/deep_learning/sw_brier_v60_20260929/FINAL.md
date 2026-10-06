# v60 SW概率平方损失结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含起点SW|最佳/停止轮|
|---|---:|---:|---|
|likelihood|73.519687911388|73.708218105973|0/18|
|brier|73.520483222623|73.708218105973|0/18|

## 探索重点与完成验证
同840771总参数/4515门控密度参数与完整初始函数；冻结v58骨干，仅门控与RQS输出更新。目标为原连续似然，或加0.5*(gate Brier+16*非default条件256格Brier)。固定canonical训练上下文是共同变化，不能把历史绝对差全部归因于辅助项。
训练578891点、验证144665点分包，Q50与原RQS公式/官方评分保持；初始源预测、参数与头哈希相同，冻结状态和真实梯度已核验。保留候选均从原始输入重载完整模型逐点验证。
沿用已修正的训练/验证分包与行映射；Brier只用训练标签按原edges映射类别，不使用验证拟合。

## 收益与不继续的点
Brier-likelihood SW +0.000000000000；训练后最高差+0.000795311235；采用None，保留[]。
初始回退不算训练收益。若Brier无配对收益，关闭当前256格平方损失系数扫描；结论不泛化所有概率评分。正局部收益仍是重复开发折证据，未经独立井确认。

## 耗时、清理和未做
{
  "candidates": [
    {
      "mode": "likelihood",
      "start": "2026-09-29T21:30:13.039963+08:00",
      "end": "2026-09-29T21:31:36.348239+08:00",
      "seconds": 78.94764420000138,
      "peak_allocated_MiB": 785.91943359375
    },
    {
      "mode": "brier",
      "start": "2026-09-29T21:30:17.826971+08:00",
      "end": "2026-09-29T21:31:39.970400+08:00",
      "seconds": 77.2769593000121,
      "peak_allocated_MiB": 785.91943359375
    }
  ],
  "cache_preparation_seconds": 5.878048499987926,
  "cache_preparation_peak_MiB": 504.87939453125,
  "pair_launch_to_completion_seconds": 86.930437,
  "first_preparation_including_failed_attempt_to_completion_seconds": 161.022657,
  "successful_preparation_to_completion_seconds": 161.022657,
  "gap_after_cache_to_first_launch_seconds": 67.041228,
  "peak_process_concurrency": 2,
  "failed_cache_attempt": false,
  "finalization_and_raw_replay_seconds": 0.3665091000002576,
  "scope": "training includes load/warmup/evaluation; cache shared once; wall times include launch/authoring gaps"
}
删除负权重与可再生缓存5项，共428257889字节。保留模型不依赖训练缓存推理，训练恢复需先重建同缓存。
目标仍是第五折严格>80.5及完整原五折OOF>82；本批无其他折、旧任务恢复或交付覆盖。未做独立新井确认。目标完成后可新目录交付。

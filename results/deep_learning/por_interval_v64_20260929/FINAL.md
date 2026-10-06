# v64 POR 容差区间质量监督结果

原第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高POR|含起点POR|最佳/停止轮|
|---|---:|---:|---|
|point_control|77.460236863284|77.551936801885|0/18|
|interval|77.454058502423|77.551936801885|0/18|

## 探索重点与已经验证
复用已经完成的v63 point对照；同18个rank8内部适配器、39888新参数、初始张量及完整函数，仅加入0.5容差区间质量负对数。区间为训练真实POR的±0.08*(abs(y)+1e-5)，原log1p归一化和Student-t3 CDF；只监督非default条件质量，原Q50和BCE/NLL保留。这是区间代理损失，不等于精确官方三角效用。
原9曲线/206输入与已有63侧特征、分折、训练井、Q、解码和评分不改。原权重冻结；实际梯度、初始和最佳144665点恢复、源哈希、匹配配置及三列评分均通过。历史控制复用不声称同步双跑或CUDA逐位重训保证。
保留者原始输入磁盘重建：[]。

## 收益与不继续
interval-control含起点POR=+0.000000000000；训练后最高POR差=-0.006178360861；采用=False，保留=False。
回退起点不是学习收益；若含起点无收益，不继续扫描当前区间宽度/辅助系数。结论仅限此POR适配器和监督配方，不泛化全部效用损失。反复使用的开发折尚无独立井确认。

## 耗时、清理和未完成
{
  "new_run_seconds": 188.48474390000047,
  "new_launch_to_completion_seconds": 192.800847,
  "historical_control_seconds": 184.9841807000048,
  "historical_control_retrained": false,
  "peak_allocated_MiB": 850.1279296875,
  "peak_process_concurrency": 1,
  "finalization_and_replay_seconds": 0.6278843999898527,
  "scope": "one new run includes loading/warmup/training/evaluation; historical control separate; authoring/finalization separate"
}
删除本轮负检查点2个；已知大小合计0字节，大小未记录2个。首次收尾在删除后遇到numpy bool JSON序列化错误，训练/评分成功；失败证据与恢复清单保留，未重训。源、配置、日志、预测保留，历史对照未改。
第五严格>80.5及完整原五折OOF>82尚未完成；无其他折、旧队列或旧交付修改。目标达成后可做独立确认与新目录交付。

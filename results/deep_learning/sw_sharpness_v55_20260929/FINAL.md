# v55整井条件网络结果

固定第五折TOTAL 80.342771761689 → 80.342771761689。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|adamw|73.489022876969|73.680963725271|0/18|
|sam|73.593073936855|73.680963725271|0/18|

## 探索重点与完成验证
同836921参数和完整v46SW初始函数；原批次点均值SW损失，全模型训练。adamw一次前反向；sam同批两遍、固定rho.05、精确恢复权重后作一次AdamW更新。优化器更新数相同，计算量不同。全部骨干/输出头/条件网络训练，不是冻结小头。原训练井隔离、Q50、1024上下文、官方解码/评分保持。
两组完整初始状态/预测相同；各自最佳恢复全144665点一致，真实训练梯度和源哈希检查完成；输入库无标签，固定参考只含64训练井。新采用目标额外磁盘重建=False，实际三列评分完成。

## 收益与方向边界
sam-adamw含回退SW差+0.000000000000；训练后最佳差+0.104051059886。采用=None。
第0轮回退不算训练收益。若SAM无匹配收益，不扫描当前扰动半径或优化器组合；结论限当前热启动/井编码与训练配置，不否定全部井条件模型。正差仍为重复开发折证据，未作独立新井确认。

## 耗时与未做
{
  "candidates": [
    {
      "mode": "adamw",
      "start": "2026-09-29T20:21:35.941791+08:00",
      "end": "2026-09-29T20:24:30.054606+08:00",
      "seconds": 170.0590813000017,
      "peak_allocated_MiB": 1249.7138671875
    },
    {
      "mode": "sam",
      "start": "2026-09-29T20:21:48.663426+08:00",
      "end": "2026-09-29T20:25:59.421848+08:00",
      "seconds": 246.17829149999307,
      "peak_allocated_MiB": 1256.1181640625
    }
  ],
  "first_launch_to_last_completion_seconds": 263.480057,
  "peak_process_concurrency": 2,
  "finalization_seconds": 0.3265591000090353,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate"
}
完整原五折OOF>82未完成；未补其他折或覆盖交付。原注意力/上采样反向可能非确定，不声称训练逐位复现。


## v55最终结果与保留边界
adamw：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
sam：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
SAM-AdamWSW=+0.000000000000；当前SW=73.680963725271，实际组合TOTAL=80.342771761689。新采用=False。
SAM无最终匹配收益，停止当前rho/优化器组合扫描；不把训练后两个退化成绩间的小差写成有效模型。
两组同836921参数、同v46SW完整起点/144665点预测；全骨干只用SW标签与Gaussian损失。两组共同原批次点均值风险；SAM同批两遍、固定rho.05、精确恢复扰动权重后更新，原Q保持，非等计算量。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[170.059, 246.178]秒；双并发总墙钟263.480秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1249.71, 1256.12]，代码与最终重建另计。
已删除4个本轮负检查点，共37951500字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v55实际更新与未采用原因
{"adamw": {"epochs": 18, "optimizer_updates": 1152, "training_forward_backward_passes": 1152, "post_update_best_epoch": 1, "post_update_best_score": 73.48902287696937, "initial_recovered": true, "first_actual_perturbation_L2": null, "last_epoch_mean_perturbed_minus_base": 0.0}, "sam": {"epochs": 18, "optimizer_updates": 1152, "training_forward_backward_passes": 2304, "post_update_best_epoch": 17, "post_update_best_score": 73.593073936855, "initial_recovered": true, "first_actual_perturbation_L2": 0.05000000074505806, "last_epoch_mean_perturbed_minus_base": 0.024398546491283923}}
SAM已实际执行两遍、固定L2扰动；两组优化器更新数相同而训练前反向数不同。训练后较对照减轻退化并不等于超过共同起点；完整选模均恢复第0轮。没有声称得到独立新井收益，也未重复旧SAM解析自检。

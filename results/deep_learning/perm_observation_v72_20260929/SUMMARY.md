# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v72 PERM有限观测分辨率

## 探索重点与对照
延续v67 RQS相对linear的局部收益+0.221226，其PERM88.827250仍低于当前89.396724。检验点似然是否追逐窄密度尖峰。本批point/interval均从v67第24轮完整838211参数和函数开始，全部参数训练；只改平均密度观测半宽0或0.05训练归一化单位。两组同五点Gauss-Legendre与logsumexp，默认BCE不变。是固定建模假设与近似积分，不宣称真实测量精度或官方效用。

## 已做与验证
两项已完成，均第18轮早停、最佳第0轮。训练后最高PERM分别88.778070945782、88.771566640436，区间组配对差-0.006504305345；含起点配对差0，没有采用。当前TOTAL80.354325927816保持。原9曲线206、原训练64井/第五16井144665点、Q50、1024中心128、原RQS解码保持；仅PERM训练标签，新AdamW3e-5/wd.01/clip2/cos80*64至1e-5/max80/pat18/min10/b64/64steps。初始状态和全点预测同一、四组真实梯度/最佳恢复/源码哈希/三列评分通过，详情见FINAL.md。

## 收益与不继续
没有本轮收益；停止当前观测宽度/积分节点扫描，不泛化所有标签噪声建模。已删除本轮4个无收益/重复权重共37411436字节，保留v67正控制、原模型与交付。实际串行峰值1，单项108.972/117.610秒、总墙钟260.439秒含启动间隔；加载预热训练评价计入，代码与收尾另计。

## 未完成与目标后
无其他折或旧队列。第五严格>80.5与完整原五折OOF>82仍未达成；接下来换机制，目标后独立井验证和新目录交付。PROJECTS/README/EXPLORATIONS/图谱已维护。
本文件修复原启动说明经PowerShell管道传输的中文编码损坏；不改已训练的源码、配置、分数和时间。


## v72最终结果
两项完成，TOTAL=80.354325927816；interval-point PERM=+0.000000000000，训练后差=-0.006504305345；采用=None，保留=[]。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见perm_observation_v72_20260929/FINAL.md。




## FINAL.md

# v72 PERM观测分辨率结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|初始PERM|训练后最高PERM|含起点PERM|最佳/停止轮|
|---|---:|---:|---:|---|
|point|88.827250136824|88.778070945782|88.827250136824|0/18|
|interval|88.827250136824|88.771566640436|88.827250136824|0/18|

## 探索与验证
同v67 RQS第24轮完整838211参数与预测，点似然或归一化目标半宽0.05的五点Gauss-Legendre平均密度负对数。两组同5节点代码，仅半宽0/.05；不是精确积分或真实观测精度假设。默认BCE、原数据/Q50/9曲线206/1024中心128/分折/解码保持，新3e-5优化器训练完整私有PERM网络。
完整初始状态和144665预测相同，实际四组梯度有效，最佳恢复、源码哈希、真实三列评分已核验。正模型磁盘重建：[]

## 收益与不继续
interval-point含起点差+0.000000000000；训练后最高差-0.006504305345。采用=None；保留=[]。
无最终配对收益则停止当前区间宽度/积分节点扫描，不泛化所有标签噪声建模。局部收益与当前冠军绝对收益区分，反复开发折不是独立泛化证明。

## 耗时与未完成
{
  "candidates": [
    {
      "mode": "point",
      "start": "2026-09-29T23:32:33.373158+08:00",
      "end": "2026-09-29T23:34:26.478428+08:00",
      "seconds": 108.97204349999083,
      "peak_allocated_MiB": 1261.5625
    },
    {
      "mode": "interval",
      "start": "2026-09-29T23:34:51.127769+08:00",
      "end": "2026-09-29T23:36:53.811832+08:00",
      "seconds": 117.61031129999901,
      "peak_allocated_MiB": 1261.5625
    }
  ],
  "first_launch_to_last_completion_seconds": 260.438674,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.35223089999635704,
  "scope": "loading/preprocessing/warmup/training/evaluation/launch gap included; authoring and finalization separate"
}
删除4负检查点共37411436字节，清单先写后删，原v67正控制和交付不改。
max80/pat18

# v72 PERM观测分辨率结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|初始PERM|训练后最高PERM|含起点PERM|最佳/停止轮|
|---|---:|---:|---:|---|
|point|88.827250136824|88.778070945782|88.827250136824|0/18|
|interval|88.827250136824|88.771566640436|88.827250136824|0/18|

## 探索与验证
同v67 RQS第24轮完整838211参数与预测，点似然或归一化目标半宽0.05的五点Gauss-Legendre平均密度负对数。两组同5节点代码，仅半宽0/.05；不是精确积分或真实观测精度假设。默认BCE、原数据/Q50/9曲线206/1024中心128/分折/解码保持，新3e-5优化器训练完整私有PERM网络。
完整初始状态和144665预测相同，实际四组梯度有效，最佳恢复、源码哈希、真实三列评分已核验。正模型磁盘重建：[]

## 收益与不继续
interval-point含起点差+0.000000000000；训练后最高差-0.006504305345。采用=None；保留=[]。
无最终配对收益则停止当前区间宽度/积分节点扫描，不泛化所有标签噪声建模。局部收益与当前冠军绝对收益区分，反复开发折不是独立泛化证明。

## 耗时与未完成
{
  "candidates": [
    {
      "mode": "point",
      "start": "2026-09-29T23:32:33.373158+08:00",
      "end": "2026-09-29T23:34:26.478428+08:00",
      "seconds": 108.97204349999083,
      "peak_allocated_MiB": 1261.5625
    },
    {
      "mode": "interval",
      "start": "2026-09-29T23:34:51.127769+08:00",
      "end": "2026-09-29T23:36:53.811832+08:00",
      "seconds": 117.61031129999901,
      "peak_allocated_MiB": 1261.5625
    }
  ],
  "first_launch_to_last_completion_seconds": 260.438674,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.35223089999635704,
  "scope": "loading/preprocessing/warmup/training/evaluation/launch gap included; authoring and finalization separate"
}
删除4负检查点共37411436字节，清单先写后删，原v67正控制和交付不改。
max80/pat18，早停实际轮数见表；未启动其他折/旧队列。严格第五>80.5与完整原五折OOF>82仍待达成；后续换思路，目标完成后独立井验证和新目录交付。



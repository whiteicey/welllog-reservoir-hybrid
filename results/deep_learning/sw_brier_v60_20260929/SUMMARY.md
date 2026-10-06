# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v60 SW概率平方损失对照

## 探索重点
从已保留v58 headstart第2轮SW73.708218开始；当前组合SW73.713976/TOTAL80.354326。v59读出扩展无最终收益，不再扫描。此次无新增模块，冻结body/norm/FiLM，只训练原gate和RQS密度，共4515参数（总840771）。两组参数、起点和样本相同，只差辅助项系数。
likelihood使用原Q加权BCE+连续NLL；brier额外加入0.5*(gate概率平方误差+16*非default条件256格Brier)。Brier为sum(p²)-2*p_true+1，与容差覆盖、分位pinball不同；类别按原归一化density_edges由训练SW标签确定，开放外侧尾部覆盖所有值。因子16固定，不扫描。它是离散概率训练目标，不声称等同官方效用或连续CRPS；保留原连续NLL。

## 数据与执行约束
原9曲线/206输入、64训练井/16验证井/原seed20261010、Q50及原解码不变。缓存冻结骨干1024上下文的中心128点，训练与验证分包；canonical固定训练上下文是两组共同变化，不将相对旧随机锚点绝对差全部归因于Brier。只读取训练包SW标签作损失；验证只选官方SW最佳。原完整起点预测及人口覆盖在缓存运行时检查，保留候选需原始输入144665点重新推理，不增加数学预检或独立审查。
新AdamW1e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。门控和密度同训练；原子保存best及last完整optimizer/scheduler/RNG/早停状态，仅显式续跑。最大2GPU，无其他折/旧队列/交付覆盖。

## 已做、收益、未做与关闭条件
已实现匹配目标，尚未训练，收益尚未证实。待共享缓存、两项训练、配对差与当前绝对分、原始输入保留模型验证、清理负权重和缓存，更新PROJECTS/EXPLORATIONS/atlas。
若Brier无最终配对收益，关闭此256格平方损失系数扫描，不泛化所有概率评分无效。第0轮回退不算学习收益。开发折重复使用，不能写成独立新井收益；完整五折OOF>82、第五严格>80.5、独立确认与新目录交付未完成。
共享缓存准备与单项训练、双并发总墙钟分别记录；含加载/预热/评估，代码间隔如实单列，不把并发耗时相加当墙钟。目标达成后可独立确认及新目录交付。


## v60最终结果
实际TOTAL=80.354325927816，Brier-likelihood SW=+0.000000000000，采用None，保留[]。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，耗时如实记录。详见sw_brier_v60_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。




## FINAL.md

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
  "successful_preparation_to_complet

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



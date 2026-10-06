# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v63 POR内部逐点与局部适配器

## 探索重点与对照
当前POR77.551936801885、组合80.354325927816；近期SW多条微调无最终收益，转回POR特征表达。沿v31 anchored完整预测器，在两专家全部18个UngatedBlock内部增加rank8适配器；原骨干、读出和路由参数全部冻结，数据与特征不变。
适配器=0.1*tanh(up(context(GELU(down(old.norm(x))))))，加在原块输出。down为C→8逐点卷积，context为8组3点深度卷积，up为8→C逐点卷积。point把3个context核系数相加后作用于同一点；local读取相邻深度。两组相同参数/初始化，所有核系数参与，不填充闲置参数；函数空间不同是对照变量。context中心权重1其余0，up全零；完整初始函数必须与当前POR逐点相同。0.1只约束单个新增块，不保证最终预测偏移有相同上界。
原v31路由的anchor参数仍冻结，但在新包装器中去掉anchor前向的no_grad，使新增上游特征的梯度经过完整路由函数；数值计算和原参数保持不变。旧源码不改。原readout使用固定尺度界的detach约定保留，不将此适配当作全骨干微调。

## 数据和训练
原9曲线/206主输入+已有63幅值侧视图，原训练64井/固定第五16井144665点，Q50/seed20261010、256随机窗口/中心128、原POR Student-t3混合+BCE损失、原256支持/800动作评分不变。只用训练POR标签，其他通道模型不更新，无外部预训练/伪标签/按验证标签选择预测。
新AdamW3e-4/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps。只优化新内部模块，各组独立选官方POR最佳，初始回退不算学习收益。旧dropout保持eval，原子完整state含适配参数/优化器/调度/RNG/早停，仅显式续跑。
鉴于v62并发OOM，本轮两项显式串行（全项目上限仍2），无自动队列/旧任务/其他折或交付覆盖。

## 已做、收益、待做
已实现两组模块与可微包装，启动前无成绩，收益尚未证实。直接训练，真实第二步检查新增参数梯度和旧参数无梯度；结束核验全部冻结状态、初始/最佳144665点预测、源哈希、实际三列评分。保留候选需原始输入磁盘重建，不重复额外预检/独立审查。
待完成两项、local-point差和当前绝对差、负权重清理、PROJECTS/EXPLORATIONS/atlas。报告独立单项耗时、实际串行总墙钟/显存，包含加载/预热/训练/评估及明确启动间隔，代码和最终验证另计。

## 不继续条件与目标
若局部访问无配对收益，不扫描本rank/核宽度/单块界限；仅限此冻结POR骨干插入位置，不泛化所有adapter。第0轮回退与重复开发折微差不能说成新井泛化。
第五严格>80.5、有明确收益后才考虑其他折；完整原五折OOF>82、新井确认和新目录交付尚未完成。目标达成后可做独立确认；现有交付保持。


## v63最终结果
两项串行完成，TOTAL=80.354325927816，local-point POR=+0.000000000000（训练后差+0.

# v63 POR内部逐点与局部适配器

## 探索重点与对照
当前POR77.551936801885、组合80.354325927816；近期SW多条微调无最终收益，转回POR特征表达。沿v31 anchored完整预测器，在两专家全部18个UngatedBlock内部增加rank8适配器；原骨干、读出和路由参数全部冻结，数据与特征不变。
适配器=0.1*tanh(up(context(GELU(down(old.norm(x))))))，加在原块输出。down为C→8逐点卷积，context为8组3点深度卷积，up为8→C逐点卷积。point把3个context核系数相加后作用于同一点；local读取相邻深度。两组相同参数/初始化，所有核系数参与，不填充闲置参数；函数空间不同是对照变量。context中心权重1其余0，up全零；完整初始函数必须与当前POR逐点相同。0.1只约束单个新增块，不保证最终预测偏移有相同上界。
原v31路由的anchor参数仍冻结，但在新包装器中去掉anchor前向的no_grad，使新增上游特征的梯度经过完整路由函数；数值计算和原参数保持不变。旧源码不改。原readout使用固定尺度界的detach约定保留，不将此适配当作全骨干微调。

## 数据和训练
原9曲线/206主输入+已有63幅值侧视图，原训练64井/固定第五16井144665点，Q50/seed20261010、256随机窗口/中心128、原POR Student-t3混合+BCE损失、原256支持/800动作评分不变。只用训练POR标签，其他通道模型不更新，无外部预训练/伪标签/按验证标签选择预测。
新AdamW3e-4/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64steps。只优化新内部模块，各组独立选官方POR最佳，初始回退不算学习收益。旧dropout保持eval，原子完整state含适配参数/优化器/调度/RNG/早停，仅显式续跑。
鉴于v62并发OOM，本轮两项显式串行（全项目上限仍2），无自动队列/旧任务/其他折或交付覆盖。

## 已做、收益、待做
已实现两组模块与可微包装，启动前无成绩，收益尚未证实。直接训练，真实第二步检查新增参数梯度和旧参数无梯度；结束核验全部冻结状态、初始/最佳144665点预测、源哈希、实际三列评分。保留候选需原始输入磁盘重建，不重复额外预检/独立审查。
待完成两项、local-point差和当前绝对差、负权重清理、PROJECTS/EXPLORATIONS/atlas。报告独立单项耗时、实际串行总墙钟/显存，包含加载/预热/训练/评估及明确启动间隔，代码和最终验证另计。

## 不继续条件与目标
若局部访问无配对收益，不扫描本rank/核宽度/单块界限；仅限此冻结POR骨干插入位置，不泛化所有adapter。第0轮回退与重复开发折微差不能说成新井泛化。
第五严格>80.5、有明确收益后才考虑其他折；完整原五折OOF>82、新井确认和新目录交付尚未完成。目标达成后可做独立确认；现有交付保持。


## v63最终结果
两项串行完成，TOTAL=80.354325927816，local-point POR=+0.000000000000（训练后差+0.004661976642）。采用=None，保留[]，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见por_internal_adapter_v63_20260929/FINAL.md。


## FINAL.md

# v63 POR内部适配器结果

第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高POR|含起点POR|最佳/停止轮|
|---|---:|---:|---|
|point|77.460236863284|77.551936801885|0/18|
|local|77.464898839926|77.551936801885|0/18|

## 探索重点与验证
两组18个rank8内部适配器，39888新参数，旧骨干/路由/头冻结；point将3个核权重用于同点，local读取邻点，同初始张量/完整函数，函数空间不同。0.1仅单块新增分支界限。
原v31锚点路由参数冻结、允许输入梯度，数值函数不变。原9曲线/206+已有63侧视图、Q50、POR Student-t3损失、分折/解码/评分保持。真实第二步新参数梯度、旧参数无梯度、冻结状态不变、初始/最佳144665点、源哈希和三列评分已核验。
保留者原始输入磁盘重建：[]。

## 收益与不继续
local-point含起点POR=+0.000000000000，训练后最高POR差=+0.004661976642；采用=None，保留=[]。
起点回退不算学习收益；若local无最终配对收益，关闭当前rank/核宽度/边界扫描，结论仅限本插入方式，不泛化所有适配器。重复开发折结果尚未经独立井确认。

## 耗时、清理与未做
{
  "candidates": [
    {
      "mode": "point",
      "start": "2026-09-29T22:00:54.015636+08:00",
      "end": "2026-09-29T22:04:03.285791+08:00",
      "seconds": 184.9841807000048,
      "peak_allocated_MiB": 850.126953125
    },
    {
      "mode": "local",
      "start": "2026-09-29T22:04:49.750125+08:00",
      "end": "2026-09-29T22:07:57.843190+08:00",
      "seconds": 183.68378750000556,
      "peak_allocated_MiB": 850.126953125
    }
  ],
  "first_launch_to_last_completion_seconds": 423.827554,
  "peak_process_concurrency": 1,
  "finalization_and_replay_seconds": 0.6049323000042932,
  "scope": "explicit serial execution after prior study OOM; individual includes load/warmup/train/eval; total includes launch gaps; authoring/finalization separate"
}
已删除负

# v63 POR内部适配器结果

第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高POR|含起点POR|最佳/停止轮|
|---|---:|---:|---|
|point|77.460236863284|77.551936801885|0/18|
|local|77.464898839926|77.551936801885|0/18|

## 探索重点与验证
两组18个rank8内部适配器，39888新参数，旧骨干/路由/头冻结；point将3个核权重用于同点，local读取邻点，同初始张量/完整函数，函数空间不同。0.1仅单块新增分支界限。
原v31锚点路由参数冻结、允许输入梯度，数值函数不变。原9曲线/206+已有63侧视图、Q50、POR Student-t3损失、分折/解码/评分保持。真实第二步新参数梯度、旧参数无梯度、冻结状态不变、初始/最佳144665点、源哈希和三列评分已核验。
保留者原始输入磁盘重建：[]。

## 收益与不继续
local-point含起点POR=+0.000000000000，训练后最高POR差=+0.004661976642；采用=None，保留=[]。
起点回退不算学习收益；若local无最终配对收益，关闭当前rank/核宽度/边界扫描，结论仅限本插入方式，不泛化所有适配器。重复开发折结果尚未经独立井确认。

## 耗时、清理与未做
{
  "candidates": [
    {
      "mode": "point",
      "start": "2026-09-29T22:00:54.015636+08:00",
      "end": "2026-09-29T22:04:03.285791+08:00",
      "seconds": 184.9841807000048,
      "peak_allocated_MiB": 850.126953125
    },
    {
      "mode": "local",
      "start": "2026-09-29T22:04:49.750125+08:00",
      "end": "2026-09-29T22:07:57.843190+08:00",
      "seconds": 183.68378750000556,
      "peak_allocated_MiB": 850.126953125
    }
  ],
  "first_launch_to_last_completion_seconds": 423.827554,
  "peak_process_concurrency": 1,
  "finalization_and_replay_seconds": 0.6049323000042932,
  "scope": "explicit serial execution after prior study OOM; individual includes load/warmup/train/eval; total includes launch gaps; authoring/finalization separate"
}
已删除负检查点4个、10051812字节；保留源码、配置、结果与删除清单。
第五严格>80.5及完整原五折OOF>82仍未完成；无其他折、旧队列或交付修改。目标达成后可做独立确认及新目录交付。



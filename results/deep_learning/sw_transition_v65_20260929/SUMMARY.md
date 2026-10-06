# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v65 SW相邻状态监督

## 探索重点与预定义对照
从当前v56 additive SW73.713975628494、组合80.354325927816出发，不继续POR容差损失或内部适配器扫描。固定detached/joint两项，同一完整SW骨干和初始函数，新增相同8257参数训练辅助头（256→32→1）。输入为相邻归一化隐藏特征的均值和绝对差，标签为两点SW default状态是否不同。辅助头不参与推理，绝不把真实状态作为模型输入。
仅改变辅助损失是否反传到骨干；detached组同样训练辅助头但detach特征，joint组联合反传。原SW主损失相同；辅助0.1*平衡BCE，正/负类各自Q加权损失均值后平均/3。只取中心128内两端均有标签且同井相邻原行的点对；井界、无效/未标注、重复填充不参与。标签仅原64训练井。

## 已做与训练边界
原9曲线/206输入、训练井拟合预处理、输入整井库不读标签/ID、原第五16井144665点、Q50、seed20261010、1024随机窗口中心128、原Gaussian+BCE/评分/解码保持。全部骨干lr3e-5，新辅助头lr3e-4；两参数组独立clip2以免detach对照因辅助梯度影响主梯度裁剪。AdamW wd.01、cosine80*64至1e-5、max80/pat18/min10/bs64/64steps。完整初始状态相同；将原函数纳入最佳选择，回退不算学习收益。
已实现，尚未训练/无成绩；实际训练中记录辅助到特征的梯度joint>0/detached=0。完整原子last_state包含网络/优化器/调度/RNG/停止状态，仅显式恢复。先串行，最近双骨干并发OOM后不把max2视作默认并发；本轮不需要双骨干，但保守控制总显存。

## 待做、收益和不继续
直接运行两项，比较最终选定SW与训练后最高SW，实际固定POR/PERM后三列评分。保留正候选前磁盘原始输入重建，源哈希/初始预测/最佳恢复作为结果证据，不增加独立预检或审查。若joint无最终配对收益，停止当前状态差辅助系数/头宽度扫描；结论只限此目标，不泛化全部辅助学习。
分别记录单项耗时与串行总墙钟，含加载/预热/训练/评估，代码和收尾另计。负权重清理前写删除清单。第五严格>80.5且明确收益后才考虑补折；完整原五折OOF>82未完成，无旧队列/其他折/交付覆盖。维护PROJECTS、EXPLORATIONS和图谱；目标完成后独立确认及新目录交付。


## v65最终结果
两项串行完成，TOTAL=80.354325927816；joint-detached SW=+0.000000000000，训练后差=-0.017791342092；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_transition_v65_20260929/FINAL.md。




## FINAL.md

# v65 SW相邻状态辅助结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|detached|73.520336553710|73.713975628494|0/18|
|joint|73.502545211618|73.713975628494|0/18|

## 探索重点与验证
同847738总参数和完整初始函数，v56完整SW模型新增8257参数训练辅助头，相邻特征均值/绝对差预测两端default状态XOR。detached组只训练辅助头，joint组将其梯度反传骨干；两者原主损失相同，辅助0.1平衡Q加权BCE/3，主/辅参数分组独立clip2；原训练井/Q50/随机1024窗口/原解码与评分不变。辅助不参与推理，标签不作输入。
完整初始状态/骨干/输出头/144665点预测相同；实际辅助特征梯度joint>0、detached=0，常规骨干/输出/FiLM及新辅助梯度有效。各自最佳恢复、源哈希和三列评分核验；保留新目标原始输入磁盘重建=False。

## 收益与不继续
joint-detached含回退SW差+0.000000000000；训练后最佳差-0.017791342092；采用=None。
回退第0轮不是学习收益。若joint无最终配对收益，不继续本状态差辅助系数或头宽度扫描；不泛化全部多任务监督。重复开发折未经独立新井确认。

## 耗时、清理与未完成
{
  "candidates": [
    {
      "mode": "detached",
      "start": "2026-09-29T22:24:40.868917+08:00",
      "end": "2026-09-29T22:26:30.644467+08:00",
      "seconds": 105.68654629999946,
      "peak_allocated_MiB": 1305.79931640625
    },
    {
      "mode": "joint",
      "start": "2026-09-29T22:26:46.861211+08:00",
      "end": "2026-09-29T22:28:37.950239+08:00",
      "seconds": 107.10151100000076,
      "peak_allocated_MiB": 1337.79931640625
    }
  ],
  "first_launch_to_last_completion_seconds": 237.081322,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.3307180999981938,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate"
}
本轮删除4

# v65 SW相邻状态辅助结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|detached|73.520336553710|73.713975628494|0/18|
|joint|73.502545211618|73.713975628494|0/18|

## 探索重点与验证
同847738总参数和完整初始函数，v56完整SW模型新增8257参数训练辅助头，相邻特征均值/绝对差预测两端default状态XOR。detached组只训练辅助头，joint组将其梯度反传骨干；两者原主损失相同，辅助0.1平衡Q加权BCE/3，主/辅参数分组独立clip2；原训练井/Q50/随机1024窗口/原解码与评分不变。辅助不参与推理，标签不作输入。
完整初始状态/骨干/输出头/144665点预测相同；实际辅助特征梯度joint>0、detached=0，常规骨干/输出/FiLM及新辅助梯度有效。各自最佳恢复、源哈希和三列评分核验；保留新目标原始输入磁盘重建=False。

## 收益与不继续
joint-detached含回退SW差+0.000000000000；训练后最佳差-0.017791342092；采用=None。
回退第0轮不是学习收益。若joint无最终配对收益，不继续本状态差辅助系数或头宽度扫描；不泛化全部多任务监督。重复开发折未经独立新井确认。

## 耗时、清理与未完成
{
  "candidates": [
    {
      "mode": "detached",
      "start": "2026-09-29T22:24:40.868917+08:00",
      "end": "2026-09-29T22:26:30.644467+08:00",
      "seconds": 105.68654629999946,
      "peak_allocated_MiB": 1305.79931640625
    },
    {
      "mode": "joint",
      "start": "2026-09-29T22:26:46.861211+08:00",
      "end": "2026-09-29T22:28:37.950239+08:00",
      "seconds": 107.10151100000076,
      "peak_allocated_MiB": 1337.79931640625
    }
  ],
  "first_launch_to_last_completion_seconds": 237.081322,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.3307180999981938,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate"
}
本轮删除4个负检查点，共38426188字节，清单先写后删除；源、配置、日志、预测与曲线保留。
实际串行，单项含加载/预热/训练/评估，总墙钟含显式启动间隔，代码和收尾另计。CUDA反向允许非确定，不声称逐位重训复现。第五严格>80.5及完整原五折OOF>82仍未完成；无其他折、旧队列或交付修改。达成后独立确认及新目录交付。



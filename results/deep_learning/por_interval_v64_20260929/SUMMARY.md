# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v64 POR物理容差质量监督

## 探索重点与固定对照
复用v63 point逐点内部适配器的已完成对照，不重复相同训练；该对照39888参数，最高含起点POR77.551936801885，训练后77.46023686328418。新候选仍是同18个rank8逐点适配器、相同初始张量和v31完整函数，仅加入训练目标。当前组合80.354325927816。
损失=原POR Student-t3混合似然+0.5*区间质量负对数。对训练非default真实POR y，用官方正收益区间半径0.08*(abs(y)+1e-5)，将左右物理边界经原log1p/median/scale映射，用原df3 CDF求16分量条件质量P；辅助=-log(max(P,1e-8))。default辅助为0，两项均原Q50加权、按所有监督点平均/3。它鼓励落入容差区间，不是官方三角效用的精确优化，也不改变验证动作/评分。原BCE/NLL保留，无验证标签训练。
原df3 CDF代码含no_grad的推理helper不能直接用于训练；新模块复用同解析公式的可微表达。POR变换是log1p、阈值8%，不套用SW的log与5%。比例0.5固定，未开展扫描。

## 边界与已做
网络完全复用v63 point：旧权重冻结、锚点权重固定且输入梯度可传播，0.1单块新增界、39888新参数；原9曲线/206+63侧视图/训练64井/第五16井/144665验证点/Q50/seed20261010/256中心128/原解码不变。AdamW3e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps。完整原子训练状态，显式续跑；本轮仅一个新任务，历史对照不恢复。
已实现候选，未有成绩。起点/新增参数哈希/优化预算与v63 point比对，运行时真实梯度/固定权重/初始与最佳144665点及三列评分验证。保留候选才额外原始输入磁盘重建，不做重复数学预检或独立审查。允许CUDA反向非确定，历史控制不是同步双跑或逐位重训保证。

## 收益、未做与不继续
收益尚未证实。待新候选训练、与已完成v63 point配对、当前绝对差、保留/清理、PROJECTS/EXPLORATIONS/atlas。若含初始选择无收益，不扫本区间宽度或辅助系数；仅限当前POR适配器与目标，不泛化所有效用监督。
单项新运行与历史对照耗时分别报告，不能把两次不同批次写成同次双并发墙钟。记录本轮实际墙钟含加载/预热/训练/评估，代码与收尾另计。无自动队列/其他折/交付覆盖，第五严格>80.5且明确收益再考虑补折；完整原五折OOF>82和独立确认未完成。目标达成后可独立验证、新目录交付。


## v64最终结果
一项新训练完成，复用v63 point对照。TOTAL=80.354325927816；interval-control POR=+0.000000000000（训练后差-0.006178360861）；采用=False，保留=False。负权重清理，来源/交付不改；无其他折或队列，目标未完成，见por_interval_v64_20260929/FINAL.md。




## FINAL.md

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





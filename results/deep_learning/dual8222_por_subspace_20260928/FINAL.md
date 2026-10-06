# POR分布子空间优化结果

观察时间：2026-09-28T09:56:37+08:00

三个对照同80.257638624609起点，只训练POR；所有其他目标/原评分保持不变。

|方案|状态|可训练参数|TOTAL|相对起点|最佳新增轮次|
|---|---|---:|---:|---:|---:|
|mean_only|完成并恢复|2064|80.268734509|+0.011095884|12|
|mean_scale|完成并恢复|4128|80.262118999|+0.004480375|6|
|density_all|完成并恢复|6192|80.258152154|+0.000513529|1|

当前已完成最佳：{'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}

每个完成结果均检查同一组144665点的实际预测与最佳参数恢复；PERM/SW逐元素不变，冻结参数没有受权重衰减影响。
新优化器和学习率调度从头开始，属于匹配热启动新实验，不冒称延续上一轮的原轨迹。

## 匹配对照
{
  "mean_only_minus_density_all": {
    "TOTAL_delta": 0.010582355239992878,
    "well_wins": 9,
    "well_losses": 7
  },
  "mean_scale_minus_density_all": {
    "TOTAL_delta": 0.003966845883679213,
    "well_wins": 8,
    "well_losses": 8
  }
}

## 误差分解
- mean_only: 10井改善、6井下降、0井不变。
{"true_default": {"n": 94357, "baseline_POR_score": 97.24556736649109, "candidate_POR_score": 97.24980658562693, "contribution_to_TOTAL_delta": 0.0008295026440396778}, "true_nondefault": {"n": 50308, "baseline_POR_score": 40.160459584663414, "candidate_POR_score": 40.258865808590826, "contribution_to_TOTAL_delta": 0.010266381598882168}}
- mean_scale: 8井改善、8井下降、0井不变。
{"true_default": {"n": 94357, "baseline_POR_score": 97.24556736649109, "candidate_POR_score": 97.17985946988564, "contribution_to_TOTAL_delta": -0.012857290982615007}, "true_nondefault": {"n": 50308, "baseline_POR_score": 40.160459584663414, "candidate_POR_score": 40.326646104565555, "contribution_to_TOTAL_delta": 0.017337665869223178}}
- density_all: 7井改善、9井下降、0井不变。
{"true_default": {"n": 94357, "baseline_POR_score": 97.24556736649109, "candidate_POR_score": 97.22649088037983, "contribution_to_TOTAL_delta": -0.00373276189817855}, "true_nondefault": {"n": 50308, "baseline_POR_score": 40.160459584663414, "candidate_POR_score": 40.201161505710346, "contribution_to_TOTAL_delta": 0.004246290901107513}}

## 后续边界
本批三个预先限定对照已经结束，不开追加宽度/LR扫描。按结果决定下一条机制；模型结果仍受反复使用第五折选模影响，完整OOF>82未完成。
epoch0若最佳，应明确为无新增训练收益。任何下一轮都应保持匹配控制，不能将增量预算或更换解码带来的差距归因于子空间。


## Final evidence and next boundary
Completed best: mean_only, TOTAL=80.268734508852, delta=+0.011095884243. POR=77.430912944379; PERM/SW unchanged. All3 runs completed; no additional training launched.
Exploratory 10000 whole-well resamples: [-0.012002,+0.034743], selection-unadjusted; not confirmatory independent validation.
Next unexecuted hypothesis: keep mixture weights/scales fixed and compare a constrained location readout against this linear mean-only control. Do not automatically open width/LR sweeps or fullCV. Full OOF>82 remains unmet. Source and prior submission unchanged.

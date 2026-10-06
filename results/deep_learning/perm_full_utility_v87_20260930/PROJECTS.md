目标：纯DL原完整OOF>82。
已做：两项实测、完整更新、恢复、实际组合、必要重载、清理。
计划/未做：转向下一明确机制；原其他折与独立泛化未做。
# v87 PERM完整网络联合效用结果

独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|PERM|固定POR/SW TOTAL|最佳/停止轮|更新后最高PERM|
|---|---:|---:|---:|---:|
|likelihood|89.396724049366|80.382093291214|0/18|89.241381002864|
|joint|89.396724049366|80.382093291214|0/18|89.257370007363|

## 探索与验证
同1324879学习参数、完整当前函数同起点；原骨干/输入路由/本目标头和小修正均实际更新，其他目标packed行仍是buffer。
原log10 Gaussian/默认0.01/官方log误差/原支持动作/分折/Q50不改；新body1e-5、head3e-5、80/18、温度.05效用0/1。实际梯度、初始144665一致、source/预处理/评分hash、buffer、最佳恢复、原行序和实际三列评价通过；正候选重载=[]。
CUDA线性上采样backward只支持warn_only确定性模式，不宣称逐位训练复现。

## 收益与不继续
最终joint-likelihood PERM +0.000000000000；排除起点的更新后最高差 +0.015989004499（诊断独立报告，非当前采用收益）。采用=None。恢复第0轮不算新增收益。
likelihood 默认/数值差={"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.24399196112769, "channel_delta": 0.0, "TOTAL_contribution": 0.0}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.13837825814508, "channel_delta": 0.0, "TOTAL_contribution": 0.0}}；实测停止=patience18
joint 默认/数值差={"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.24399196112769, "channel_delta": 0.0, "TOTAL_contribution": 0.0}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.13837825814508, "channel_delta": 0.0, "TOTAL_contribution": 0.0}}；实测停止=patience18
无绝对或匹配最终收益时关闭本源/全网络/低LR/系数1/温度.05组合，不继续临时扫效用或延长预算；结论不泛化所有PERM训练。
一个反复开发的第五折不是完整OOF或独立泛化证明。

## 成本、清理、未完成与目标后
{
  "candidates": [
    {
      "mode": "likelihood",
      "start": "2026-09-30T12:48:03.378445+08:00",
      "end": "2026-09-30T12:53:07.288306+08:00",
      "seconds": 298.9389990000054,
      "peak_allocated_MiB": 917.45361328125,
      "peak_reserved_MiB": 1022.0,
      "launch_to_completion_seconds": 303.909861
    },
    {
      "mode": "joint",
      "start": "2026-09-30T12:51:25.899575+08:00",
      "end": "2026-09-30T12:56:33.055661+08:00",
      "seconds": 301.050231399975,
      "peak_allocated_MiB": 917.45361328125,
      "peak_reserved_MiB": 1022.0,
      "launch_to_completion_seconds": 307.156086
    }
  ],
  "first_launch_to_last_completion_seconds": 509.677216,
  "first_launch_to_verified_seconds": 722.222559,
  "process_overlap_seconds": 101.388731,
  "peak_training_processes": 2,
  "sum_individual_launch_seconds": 611.065947,
  "finalize_seconds": 0.40619530002004467,
  "scope": "input loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring and atlas separate; not a capacity benchmark"
}
删除4个冗余权重共57604292字节，正源和旧交付不改。
未运行其他折、旧队列。仍需原第五严格>80.5且明确收益再考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。

## 收尾与结论范围
更新后最高的+0.015989004499发生在似然第1轮/效用第2轮，不能写成同一轮差。18个同轮比较只有6轮效用更高、12轮更低，平均差-0.002843593202，最后一轮-0.019259906293。因此存在最高值的小正诊断，但没有一致轨迹改善或采用收益，当前完整私有微调设置不继续扫描。源/config/学习曲线保留；负权重4个57604292字节已按清单清理。图谱632节点、87个dead end，主节点显示原第五实际固定POR/SW TOTAL80.382093291214及恢复起点，不冒称新收益。原正源/交付保留，没有GPU训练或其他折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

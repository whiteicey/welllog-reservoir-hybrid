# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md



## v35门控失活诊断
2026-09-29T14:17:26.948609+08:00。已保存第18轮状态在64固定训练锚点7973有效点上，matched/shuffled门控proposal均100%负值，alpha与梯度全部为0；初始output梯度非零，但query/key零。18轮验证分数不变。不能将此结果当作充分训练后的检索负证据。旧源码/权重不改，新v36仅修复平滑门控通路。




## FINAL.md

# v35 收尾：门控失活

原两项均第18轮完成，全部验证分数保持73.6630827970656。后续末态检查确认硬截断门控全关闭且梯度为0；不能将其作为充分训练后的跨井检索负证据。详见gate_diagnostic.json。历史源和checkpoint未改。v36用平滑门控作独立修正，详见../retrieval_smooth_gate_v36_20260929/FINAL.md。





# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v184 原SOURCE私有PERM骨干与正损失协同

## 探索重点
保留174已实测的SAM0.05/conditionalRPS1/效用0.1，开放本PERM副本原双专家body更新。对照174PERM89.364985921273：相同Source初始预测/loss/数据/Q/预算，只扩大更新范围；head3e-4、body1e-5，原输出行/aux/router参数固定但gate/scale函数随特征变化。

## 已探索与验证
已实现，未训练无分数。只复用166结构/捕获梯度实现，不加载其权重；166的旧jointRPS+AdamW no gain不是本正损失组合的结论。Source checkpoint不修改，其他POR151/SW104模型独立且不更新。80-14/min10/b64/64步/cos80*64，body保持eval关闭dropout但允许梯度。

## 收益
尚未证实。对照active82720与本1355414不同，参数更新范围/SAM梯度范数/计算改变，不冒称同活动容量消融。主训练只读取PERM标签；无teacher/预测投票/新的第六阶段。最终实际组合144665点后报告局部与绝对15489.405564856919对照。

## 不继续的点
不同时改变family/采样/teacher或扫body学习率；未有益关闭此组合，不能用旧166负结果提前认定本实验失败。严格TOTAL>80.8有收益后补折，原OOF>82未完成。

耗时待实测：body/backbone额外训练和SAM两遍，单项/总工作墙钟/显存/数据/预热分开，不当全方法成本。


Live launch confirmed: {"process_verified_live": true, "pid": 89832, "state": {"status": "running", "pid": 89832, "stage": "perm_conditional", "epoch": 9, "best": 89.3447601747529, "best_epoch": 0, "bad": 9, "peak_MiB": 1055.93701171875}, "normal_GPU": true, "budget": "80/14/min10/b64/64steps", "source_counted_total_stages": 5, "complete_five_folds": false, "remaining_folds_started": false, "result_scope": "live training snapshot, not final gain"}. v185 repairs missing functional import only; random width128 differs in hidden initialization/capacity from width96 control, zero-output train-prior initial predictive function is shared. v184 freezes original head/router parameters but allow

# v184 原SOURCE私有PERM骨干与正损失协同

## 探索重点
保留174已实测的SAM0.05/conditionalRPS1/效用0.1，开放本PERM副本原双专家body更新。对照174PERM89.364985921273：相同Source初始预测/loss/数据/Q/预算，只扩大更新范围；head3e-4、body1e-5，原输出行/aux/router参数固定但gate/scale函数随特征变化。

## 已探索与验证
已实现，未训练无分数。只复用166结构/捕获梯度实现，不加载其权重；166的旧jointRPS+AdamW no gain不是本正损失组合的结论。Source checkpoint不修改，其他POR151/SW104模型独立且不更新。80-14/min10/b64/64步/cos80*64，body保持eval关闭dropout但允许梯度。

## 收益
尚未证实。对照active82720与本1355414不同，参数更新范围/SAM梯度范数/计算改变，不冒称同活动容量消融。主训练只读取PERM标签；无teacher/预测投票/新的第六阶段。最终实际组合144665点后报告局部与绝对15489.405564856919对照。

## 不继续的点
不同时改变family/采样/teacher或扫body学习率；未有益关闭此组合，不能用旧166负结果提前认定本实验失败。严格TOTAL>80.8有收益后补折，原OOF>82未完成。

耗时待实测：body/backbone额外训练和SAM两遍，单项/总工作墙钟/显存/数据/预热分开，不当全方法成本。


Live launch confirmed: {"process_verified_live": true, "pid": 89832, "state": {"status": "running", "pid": 89832, "stage": "perm_conditional", "epoch": 9, "best": 89.3447601747529, "best_epoch": 0, "bad": 9, "peak_MiB": 1055.93701171875}, "normal_GPU": true, "budget": "80/14/min10/b64/64steps", "source_counted_total_stages": 5, "complete_five_folds": false, "remaining_folds_started": false, "result_scope": "live training snapshot, not final gain"}. v185 repairs missing functional import only; random width128 differs in hidden initialization/capacity from width96 control, zero-output train-prior initial predictive function is shared. v184 freezes original head/router parameters but allows body gradients; fixed POR/SW remain unchanged.

V184 final PERM=89.350304704679, compactcontrolgain=-0.028089506464; actualTOTAL=80.593134364710, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=969.181, peakMiB=1055.937; all144665replay/frozen exact.


Final actual PERM89.350304704679, best/stop10/24; opening body versus matched positive174 -0.014681216593, no body-scope gain. Same-active-capacity166 secondary training-combination gain +0.005544529926, changed SAM/RPS/utility together, not isolated contribution attribution; retain best/last as secondary positive control. ActualTOTAL80.593134364710,144665 exact replay/frozen verified; current154 unchanged. Close this body-update route, no LR/body grid. Single training969.180803s, peak1055.937MiB, staggered185 then186 overlap, not equal-compute or full5-stage time.



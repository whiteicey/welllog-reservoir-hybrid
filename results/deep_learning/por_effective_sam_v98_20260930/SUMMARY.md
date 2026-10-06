# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v98 POR有效模块的优化器对照：最终记录

探索重点：从v96固定组合TOTAL80.504197424477的POR来源v31 anchored完整函数热启动，只联合更新现有路由与幅值位置修正。预定义AdamW与SAM(rho0.05)同容量43635参数、同初始权重、同训练窗口、学习率3e-5的对照。原双骨干、其他密度因子和锚点冻结，主损失为自身POR Student-t3混合NLL+BCE/Q50。

已探索与验证：两组实际训练至第18轮，patience18停止，均恢复第0轮原模型。完整144665点恢复一致，冻结状态逐张量一致，实际训练梯度覆盖路由/位置，参数实际更新。AdamW更新后最佳为第1轮TOTAL80.502927440098，SAM第1轮80.503227187203。不存在完整五折证据，也未启动本候选其他折。

取得收益：相对v96 TOTAL均为0；预定义选定SAM减AdamW消融差0。训练损失有所下降但官方验证成绩未提高。80.504197仍来自保留的第0轮父函数，不属于本次新收益，positive_gain_and_gate两项均false。耗时256.761939秒，峰值719.845703MiB；包括加载、预热、初始全点评估、训练、验证与保存。两组在同一GPU进程共享冻结特征、串行更新，和v97补折任务重叠；无同等工作量串行耗时对照，不声称加速。

不继续的点及证据：当前热启动/43635参数/学习率3e-5/SAM0.05两项均回退原函数，关闭此配置邻域的rho/LR扫描。结论仅限本固定对照，不泛化为SAM或POR优化整体无效。下一机制可测试评分目标对齐，须独立记录。

数据与训练边界：原9曲线、206维与既有幅值63侧特征、原第五折/seed20261010/Q50/原单覆盖解码；80/18/min10/batch64/64步、cosine80*64/eta_min1e-5。最大2GPU任务；原模型、交付与v97固定补折配方不变。

保留新源码、配置、日志、历史、梯度/初始/最终证明与预测。两组冗余第0轮权重、负结果更新权重和last-state按deletion_manifest.json记录删除；原v31/v96正来源不动。不能再对本负实验自动恢复。





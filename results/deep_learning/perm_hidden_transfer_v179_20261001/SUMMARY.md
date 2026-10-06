# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v179 私有PERM训练侧隐藏迁移

## 探索重点
在177井均衡+完整骨干SAM的预测结构上，增加训练侧128→256辅助投影，匹配原本折SOURCE两专家隐藏特征。假设冷启动私有骨干可吸收已有训练井表征，减少泛化损失。只用64训练井输入窗口，不使用教师预测/新标签/外部权重。
对照177仍运行，最终结果待确认；现有旧167自身骨干锚定0增益并不等同冷启动表征迁移；历史132曾有局部迁移收益，不能直接转为本实验收益。

## 已探索与验证
已实现，未训练无成绩。预测网络685873参数不变，另33024训练辅助参数；教师SOURCE冻结，已在全方法阶段1计数，不新预训练。NLL+BCE+jointRPS1和采样/LR6e-4/80-14/seed保持，增加0.1归一化隐藏MSE，Q/核心窗口掩码同预测损失。SAM全梯度扰动包含aux，新活动容量不是与177完全相同，辅助参数及计算影响需如实报告。
验证/eval从不运行教师、不进入教师输出，辅助头不进入最终预测。初始student由同随机seed构造，原SOURCE仅教师；最终实际组合144665点。

## 收益
尚未证实；控制预测容量同、训练辅助容量+33024，不能冒称同active参数消融。不得加载177权重增加第六阶段。

## 不继续的点
若无收益，不扫隐藏系数、teacher层或蒸馏预测；保持输出私有、无验证标签拟合、原解码不改。绝对v15489.405564856919仍对照，严格TOTAL>80.8且有收益才补折；OOF>82未完成。

耗时：待实测；两次SAM前后向加冻结教师计算，训练单项/同等总工作墙钟、显存/数据/预热如实区分，不称全5阶段成本。

启动前对照177已最终确认PERM88.970890837572，相对175 +0.077351289783；本项尚未训练无分数。教师只在训练forward运行，eval不调用，额外aux训练参数影响SAM梯度范数如实记录。

v179显式启动PID5808，normal GPU/临时SYSTEM_REQUIRED。训练快照{'status': 'running', 'pid': 5808, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.64084730011055, 'best_epoch': 12, 'bad': 0, 'peak_MiB': 823.6640625}，active718897，非最终收益；目前最多2任务、全方法仍5阶段，未补折。
初始student PERM68.383772436967与177一致；教师仅train forward计算隐藏目标，eval不运行教师。预测685873/aux33024、active718897；原SOURCE冻结、无新预训练/教师预测损失/教师推理。CUDA插值和attention反向warn_only非确定性警告同对照，未报运行错误，不声称位级训练复现。

V179 final PERM=88.881649640932, compactcontrolgain=-0.496744570212; actualTOTAL=80.429105092399, fivephases inclsource;strict80

# v179 私有PERM训练侧隐藏迁移

## 探索重点
在177井均衡+完整骨干SAM的预测结构上，增加训练侧128→256辅助投影，匹配原本折SOURCE两专家隐藏特征。假设冷启动私有骨干可吸收已有训练井表征，减少泛化损失。只用64训练井输入窗口，不使用教师预测/新标签/外部权重。
对照177仍运行，最终结果待确认；现有旧167自身骨干锚定0增益并不等同冷启动表征迁移；历史132曾有局部迁移收益，不能直接转为本实验收益。

## 已探索与验证
已实现，未训练无成绩。预测网络685873参数不变，另33024训练辅助参数；教师SOURCE冻结，已在全方法阶段1计数，不新预训练。NLL+BCE+jointRPS1和采样/LR6e-4/80-14/seed保持，增加0.1归一化隐藏MSE，Q/核心窗口掩码同预测损失。SAM全梯度扰动包含aux，新活动容量不是与177完全相同，辅助参数及计算影响需如实报告。
验证/eval从不运行教师、不进入教师输出，辅助头不进入最终预测。初始student由同随机seed构造，原SOURCE仅教师；最终实际组合144665点。

## 收益
尚未证实；控制预测容量同、训练辅助容量+33024，不能冒称同active参数消融。不得加载177权重增加第六阶段。

## 不继续的点
若无收益，不扫隐藏系数、teacher层或蒸馏预测；保持输出私有、无验证标签拟合、原解码不改。绝对v15489.405564856919仍对照，严格TOTAL>80.8且有收益才补折；OOF>82未完成。

耗时：待实测；两次SAM前后向加冻结教师计算，训练单项/同等总工作墙钟、显存/数据/预热如实区分，不称全5阶段成本。

启动前对照177已最终确认PERM88.970890837572，相对175 +0.077351289783；本项尚未训练无分数。教师只在训练forward运行，eval不调用，额外aux训练参数影响SAM梯度范数如实记录。

v179显式启动PID5808，normal GPU/临时SYSTEM_REQUIRED。训练快照{'status': 'running', 'pid': 5808, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.64084730011055, 'best_epoch': 12, 'bad': 0, 'peak_MiB': 823.6640625}，active718897，非最终收益；目前最多2任务、全方法仍5阶段，未补折。
初始student PERM68.383772436967与177一致；教师仅train forward计算隐藏目标，eval不运行教师。预测685873/aux33024、active718897；原SOURCE冻结、无新预训练/教师预测损失/教师推理。CUDA插值和attention反向warn_only非确定性警告同对照，未报运行错误，不声称位级训练复现。

V179 final PERM=88.881649640932, compactcontrolgain=-0.496744570212; actualTOTAL=80.429105092399, fivephases inclsource;strict808=False, no otherfolds. best/stop=21/35, seconds=767.048, peakMiB=823.664; all144665replay/frozen exact.

最终隐藏迁移负结果：PERM88.881649640932，相对177 -0.089241196640；best/stop21/35，actualTOTAL80.429105092399。预测结构685873不变，aux训练+33024、teacher仅训练、全局SAM梯度范数改变，非完全同active容量消融。144665最佳恢复/冻结验证通过，负权重清理留SHA，原SOURCE及177正结果保留。关闭本次0.1两专家hidden监督，不扫系数/层，不泛化所有表征迁移。



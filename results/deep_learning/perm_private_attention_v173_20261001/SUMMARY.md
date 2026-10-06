# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v173 独立PERM深度注意力骨干

## 探索重点
从头训练完整私有PERM：96维TCN、16点池化token、四层注意力、128维局部细化，16Gaussian+default。原SOURCE只用于固定POR的阶段计数，不进入PERM网络。假设：原冻结先验限制状态与密度协调，新骨干可联合改进。

## 已探索与验证
已实现，未启动、无成绩。原206输入+相对深度，无新数据。gate先验、连续分量分位数/尺度仅578891训练锚点拟合。自身NLL+BCE+RPS1，lr6e-4，80/14/min10/b64/64步。以全PERM评分恢复最佳；三头实际组合144665点。

## 收益
尚未证实。绝对对照v15489.405564856919；随机初始化与原来源热启动并非同初始化消融，不将差值归因于单一结构。总方法仍5阶段。

## 不继续的点
没有收益不做大规模宽度/注意力层数扫描；初始化不是验证标签拟合，不更改评分解码。旧输出先验修正路线172在14轮无收益；不加载其权重。

耗时记录区分本项和全方法总墙钟，启动与171错峰重叠，warmup未另跑，数据及初始/最终评价计入，最终组装另计。

训练快照（非最终）：v173 {'status': 'running', 'pid': 64464, 'stage': 'perm_conditional', 'epoch': 4, 'best': 87.79233514602183, 'best_epoch': 3, 'bad': 1, 'peak_MiB': 788.1005859375}；active685873。尚无最终磁盘恢复和三头组合，不改变当前配方。
CUDA插值及memory-efficient attention反向有warn_only非确定性警告，未报运行错误；不声称训练轨迹位级可复现，最终恢复预测一致性仍需验证。初始化68.383772436967，前2轮86.192730/87.420202；这是独立随机模型学习进展，非超过热启动对照的收益。

V173 final PERM=88.847573085146, compactcontrolgain=-0.530821125998; actualTOTAL=80.417178297873, fivephases inclsource;strict808=False, no otherfolds. best/stop=20/34, seconds=286.037, peakMiB=788.101; all144665replay/frozen exact.

最终独立骨干：PERM88.847573085146，相对v154 -0.557991771773，实际TOTAL80.417178297873；best/stop=20/34，active685873，144665磁盘恢复一致，训练286.037s，peak788.101MiB。虽然随机初始到学习后提升，但未超过来源/有效头、没有同容量消融正证据；不作为有效增益。清理负权重留SHA；本轮冷启动注意力骨干停止延伸，不泛化所有骨干。





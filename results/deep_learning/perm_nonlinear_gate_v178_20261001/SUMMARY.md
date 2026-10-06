# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v178 非线性默认状态读出

## 探索重点
在174的SAM0.05/conditionalRPS1/分箱效用0.1和128维位置/权重读出上，增加33号输出作默认logit修正，tanh界限1。读取同319维双专家联合隐藏/幅值特征；原source gate权重、scale、骨干均固定。假设分布读出与默认状态的条件信息需要联合学习。
对照174PERM89.364985921273；绝对15489.405564856919。旧161/168只训练原线性gate行，机制不相同。

## 已探索与验证
已实现、未训练无成绩。新输出零初始化，source初始函数保持；active82849，比174多129，其他hidden层参数初始化保持、不是等active容量消融。自身NLL+BCE/条件CDF/评分效用，80-14/min10/b64/64步/cos80*64，固定原fold5/seed/Q/评分解码。不得加载174训练头增加第六阶段。

## 收益
尚未证实；必须最终恢复并实际组合144665点，报告局部对照差值和绝对分。控制范围+129参数改变，不冒称同参数数目证明纯机制。

## 不继续的点
本项不同时增加尺度修正、改采样/效用或扫gate界限、隐层宽度。未超过80.8不补其余折，OOF>82未完成；旧交付不改。

耗时待实测：单项和总工作墙钟分开，data/初始/最终评价包含、不另预热，GPU最大2、峰值待记录。

v178显式启动PID7844，normal GPU/临时SYSTEM_REQUIRED。训练快照{'status': 'running', 'pid': 7844, 'stage': 'perm_conditional', 'epoch': 4, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 4, 'peak_MiB': 786.46533203125}，active82849，非最终收益；目前最多2任务、全方法仍5阶段，未补折。

V178 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=373.289, peakMiB=786.465; all144665replay/frozen exact.

最终非线性状态支路无收益：PERM89.344760174753，相对174 -0.020225746520；best/stop0/14，actualTOTAL80.591193779236，active+129非等容量；144665恢复/冻结验证通过。同Source初始函数/预算/Q/解码，关闭本轮共享trunk、±1gate支路，不扫界限/宽度，不泛化所有非线性gate；负权重清理留SHA。





# v182 PERM Student-t5分量

## 探索重点
保留174的128维位置/权重读出、SAM0.05/conditionalRPS1/分箱效用0.1，Gaussian改Student-t5。每分量标准化log10均值/方差匹配：t尺度=原sd*sqrt(3/5)。假设更稳健尾部梯度可改善PERM少数大误差。

## 已探索与验证
已实现、未训练无分数。PDF与闭式CDF共同用于NLL、RPS、utility和预测；原256格/800动作/评分/分折/Q/seed/80-14/b64/64步不动。保留原Source参数作为起点，不加载174头。初始CDF/预测不同；不能称完全同初始函数消融。

## 收益
尚未证实。同82720参数、预算、优化方法，报告相对自己初始分与Gaussian17489.364985921273的最终差值；绝对15489.405564856919。匹配的是log10空间分量矩，不声称物理PERM矩有限/匹配。总5阶段，实际144665三头验证后才判断收益。

## 不继续的点
本项只固定df5，不扫自由度/Decoder动作/网格分辨率；此前Laplace负结果不能预先推定t5无效。无益关闭此族设置，不添加第六阶段、其他折或覆盖交付。严格TOTAL>80.8有收益才补折，OOF>82未完成。

耗时待实测：数据/初始/最终评价包含、无另预热，单项/总墙钟及并发显存区分，不当成全方法总成本。

v182已显式启动PID85420、normal GPU、临时SYSTEM_REQUIRED。当前训练快照{'status': 'running', 'pid': 85420, 'stage': 'perm_conditional', 'epoch': 5, 'best': 89.33653883690049, 'best_epoch': 0, 'bad': 5, 'peak_MiB': 786.46533203125}，active82720，非最终收益；max2任务/完整方法5阶段/未补折。
初始PERM89.33653883690049，Gaussian原SOURCE89.3447601747529；CDF变化导致初始函数不同，报告最终score和对各自初始的变化，不能把初始PDF差异当作训练增益。

V182 final PERM=89.336538836900, compactcontrolgain=-0.041855374243; actualTOTAL=80.588316310987, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=327.517, peakMiB=786.465; all144665replay/frozen exact.

最终t5分布无益：PERM89.336538836900，相对174 -0.028447084372；自己初始89.336538836900、学习增益0、best/stop0/14，actualTOTAL80.588316310987。参数/优化/预算/Q一致但初始CDF不同、log分量矩匹配；PDF/CDF/utility/decoder同族，144665恢复一致/冻结通过。负权重清理留SHA；保留Gaussian，不扫df或混合族，结论仅限本df5设置。

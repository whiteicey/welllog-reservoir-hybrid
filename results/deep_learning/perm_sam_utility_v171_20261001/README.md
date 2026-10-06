# v171 PERM有限探索

## 探索重点
SAM/RPS加PERM log10误差三角效用，系数0.1。
假设：改善PERM连续低估/分布质量；固定来源初始化、分折、种子、Q50、评分解码；对照v165 SAM PERM89.35224836888084，绝对对照v15489.40556485691894。

## 已探索与验证
已实现，尚未训练；没有成绩。只替换PERM阶段，固定POR151/SW104。最大80/patience14/min10，bs64/64步，cos80*64。各自优化器和最佳官方PERM恢复；完成后实际组合144665点。

## 收益
尚未证实；SAM新增效用只有训练侧标签，效用代理不等于解码argmax。新TCN与旧读出容量及输入不同，不能把分差归因于单一因子。

## 不继续的点
若无收益不扫描本轮效用系数/TCN宽度或膨胀率。无第六阶段、其他折或交付覆盖。

耗时：待实测；并发最多2，记录单项与同等工作总墙钟、显存、数据及预热边界。

已显式启动PID39440，normal GPU；来源函数初始PERM89.344760174753，可训练参数82720，80/14。当前快照{'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 1, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}；尚未最终恢复，收益尚未证实。
训练效用系数0.1记录于initialization.train_only_expected_logratio_utility_weight；config.utility_weight是继承的旧SW alpha占位0，本实验实际PERM损失为NLL+BCE+RPS1+0.1效用。SAM半径0.05，2次前后向。

训练快照（非最终）：v171 {'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 14, 'best': 89.36208510714584, 'best_epoch': 10, 'bad': 4, 'peak_MiB': 786.46533203125}；active82720。尚无最终磁盘恢复和三头组合，不改变当前配方。

V171 final PERM=89.362085107146, compactcontrolgain=-0.016309103998; actualTOTAL=80.597257505573, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=588.786, peakMiB=786.465; all144665replay/frozen exact.

最终局部正收益：PERM89.362085107146，同容量/初始化/80-14/SAM半径/步数/Q/评分对照v165差值+0.009836738265，但比v154 -0.043479749773。实际TOTAL80.597257505573，best/stop=10/24；144665恢复一致与冻结核验通过。保留best/last正结果。额外效用CDF计算使等步数不等计算量；不声称完整OOF收益，不替换有效配方。

本批171/172/173三次探索总墙钟598.674s（错峰最多2GPU，包含数据与初始/最终评价，不含组装，不是完整5阶段训练时间）；逐项秒数和显存见batch_timing.json。

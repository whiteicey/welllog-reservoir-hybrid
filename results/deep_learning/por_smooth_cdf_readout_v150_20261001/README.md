# v150 平滑累计概率分布读出

探索重点：以条件单调CDF变形替代逐区间自由密度修正。设原条件CDF为F，使用8个Beta(i+1,8-i)累计分布的条件混合W，输出新CDF W(F(y))，新密度为p(y)W'(F(y))。训练和推断使用同一函数及解析导数；8个系数限制logit±0.5。均匀系数时W(u)=u；实现残差形式使初始变形严格为零。

已经探索与验证：实现完成，尚未启动、没有成绩。256隐层仍输出16个位置与16个log权重增量，另加8个CDF系数，共233000训练参数，相对v140增加2056，不是匹配容量消融。原父模型v113冻结，末层全零；不加载训练过的v140/v148/v149头。自身真实POR非默认标签只进入似然，CDF在标签处解析求值；没有标签侧输入、伪标签或解码后拟合。BetaCDF由8次二项式尾和解析计算，无数值分位点反演。

收益：尚未证实；绝对对照POR77.912102888582。恢复最佳后实际与固定PERM136/SW104三通道合并，不用其他目标真值代替预测。保持原区间/动作网格/默认门控/原混合族及参考尺度，不调分辨率或阈值。新增CDF概率空间基函数数量不等于新增解码网格。

不值得继续的点：逐区间有界密度v148完成POR77.889012，低于256对照0.023091；v145尺度修正和192/384/512宽度无收益。本方案改为平滑低维分布形状，不延续那些配置的边界/学习率扫描。

约束：原9曲线、206+63幅值、train-only64井统计、Q50、原第五折、seed20261010、自身似然与官方评分保持。normal/max2、80/18/min10/b64/64步、3e-4、cos80*64；完整原子last与显式恢复、进程级防睡眠。source+POR113+POR150+PERM136+SW104共5阶段，未达第五折严格TOTAL>80.8与收益前不补其他折。完整OOF>82未完成，旧交付不变。

待完成：GPU槽位空出后显式启动，记录初始成绩、真实训练/墙钟/并发/峰值；最佳磁盘恢复、144665覆盖及冻结完整性、实际合并与最终结论；负权重清理留SHA清单。此候选不意味着自动后续搜索。

Explicit launch PID68944 verified command line and actual epoch2 training after v149 both complete. Initial original-fifth POR77.540454326739 identical to frozen source;233000activeparameters; peak924.940MiB, oneGPU, noOOM or stderr. Initial progress is not gain; final best restoration/assembly pending. No remainingfolds or new automatic candidates.

V150 bernstein8: restored POR=77.769405183474; delta vs v114=+0.221335688581, vs historical=+0.217468381589; actual fifth TOTAL=80.559212678422;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=248.118, peakMiB=924.940.

V150 final POR77.769405183474/TOTAL80.559212678422; delta vs effective256 -0.142697705108; best17/stop35. Exact144665/frozen replay, actual three-target assembly. Training seconds248.118, peakMiB924.940; launch/report wall and overlap in timing.json. Negative weights removed with SHAmanifest. Close thisCDF shape config; latestuser directs augmentation of proven gains, v151same230944parameter256readout plus jointRPS launched, source-counted5phases. No otherfolds; bestv140 unchanged.

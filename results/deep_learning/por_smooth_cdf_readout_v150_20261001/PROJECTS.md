# v150 进展

目标：从平滑分布读出获得第五折增益，最终完整OOF>82。

已完成：8项Beta混合单调CDF与密度导数实现；原函数零初始化；自身似然及原网格推断同步；PLAN定义。

计划/未完成：两项v149尚在训练，候选暂未启动。空位后显式启动，恢复最佳、实际三通道合并、记录收益/时间/显存/负证据并更新图谱。

目标完成后可选：POR证实增益后分别转移其他目标；当前不补折、不启动额外机制。

Explicit launch PID68944 verified command line and actual epoch2 training after v149 both complete. Initial original-fifth POR77.540454326739 identical to frozen source;233000activeparameters; peak924.940MiB, oneGPU, noOOM or stderr. Initial progress is not gain; final best restoration/assembly pending. No remainingfolds or new automatic candidates.

V150 bernstein8: restored POR=77.769405183474; delta vs v114=+0.221335688581, vs historical=+0.217468381589; actual fifth TOTAL=80.559212678422;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=248.118, peakMiB=924.940.

V150 final POR77.769405183474/TOTAL80.559212678422; delta vs effective256 -0.142697705108; best17/stop35. Exact144665/frozen replay, actual three-target assembly. Training seconds248.118, peakMiB924.940; launch/report wall and overlap in timing.json. Negative weights removed with SHAmanifest. Close thisCDF shape config; latestuser directs augmentation of proven gains, v151same230944parameter256readout plus jointRPS launched, source-counted5phases. No otherfolds; bestv140 unchanged.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

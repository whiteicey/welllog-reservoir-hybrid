# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v180 井均衡机制迁移到PERM读出

## 探索重点
将175在独立骨干上实测+0.045966的50%行/50%井采样，迁移到174已证实条件CDF+SAM+效用读出。对照174PERM89.364985921273，同82720参数/原SOURCE初始化/ownloss/Q/80-14/原解码，唯一变化为训练锚点采样。

## 已探索与验证
已实现，未训练无分数。采样只使用64训练井锚点分组，不用标签/验证设权、不输入井ID。相同seed20261010但采样窗口序列不同；不加载174或175训练权重。SAM0.05、RPS条件连续1、分箱效用0.1、lr3e-4/min10/b64/64步/cos80*64保持。

## 收益
尚未证实，骨干上的消融收益不能提前算成读出收益。须最终恢复并实际三头组合144665点，报告配对delta和绝对89.405565对照。总5阶段来源只计一次。

## 不继续的点
本项不同时改非线性gate/尺度或CDF效用，不扫描采样比例。无收益关闭这次迁移，未超过严格TOTAL80.8不补折，原OOF>82未完成。

耗时待实测：单项/全工作墙钟、并发和显存区分，数据/初始/最终评价包含，不另预热；组装另计。

V180 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=333.678, peakMiB=786.465; all144665replay/frozen exact.

最终井均衡迁移到读出无益：PERM89.344760174753，相对174 -0.020225746520；best/stop0/14，actualTOTAL80.591193779236，同容量/初始化/loss/优化/Q/解码核验通过，144665恢复一致。清理负权重留SHA；保留175骨干原采样正收益，但不跨架构认定采样泛化；不扫比例、读出继续用174原行采样。





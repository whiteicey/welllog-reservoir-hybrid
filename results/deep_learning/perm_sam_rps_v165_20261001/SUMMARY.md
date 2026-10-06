# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM v165：有效RPS读出上的SAM匹配对照

## 探索重点

保留有效Gaussian128位置/权重/NLL/BCE/RPS1，改优化器扰动更新SAM rho0.05。既有SW的SAM局部收益仅为动机，不能声称PERM已有效。两候选adamw/sam，同参数82720和同初始函数，80/14；真实新AdamW控制避免历史18与14预算混淆。

## 已探索内容与验证

准备实现完成，尚未启动两项、无分数。复用已有SAM双前后向、RNG回放、参数clone精确回滚、每组clip2；只改rho0.05，与无扰动AdamW对照。两项独立SOURCE起点，无已训154头；同样训练窗口和64更新/轮。SAM每更新两前后向、控制一前后向，等更新预算不等计算预算，分别记录单项时间及完成两项总墙钟/并发/显存/数据预热计入。固定POR151/SW104实际合并，五阶段。

## 收益

尚未证实，预定义sam−adamw同预算机制差值和对历史正PERM89.405564856919的绝对差值。未完成两项不声称最终消融收益。

## 不值得继续的点及证据

低LR163在14轮无改善，不继续LR网格；尺度/Laplace/gate/适配局部配置已关闭，结论仅限实测配置。此项无收益不继续SAM半径网格。


## 显式启动控制项

POR164 linear已完成，暂保留权重至配对比较结束；非线性仍运行。已用空槽显式启动165 AdamW控制，SAM尚未启动，无第三GPU任务或自动队列。私有骨干166已准备但尚无训练/分数，等待现有预定义对照。

V165 adamw: actual PERM=89.344760174753, TOTAL=80.591193779236; delta historicalRPS=-0.060804682166; best/stop=0/14, seconds=131.987, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 sam: actual PERM=89.352248368881, TOTAL=80.593814647181; delta historicalRPS=-0.053316488038; best/stop=10/24, seconds=397.458, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 completed matched80/14 pair: SAM-AdamW PERM=+0.007488194128. Equal updates, unequal compute; original fifth only, not fullOOF.


Completed adamw: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.59119377

# PERM v165：有效RPS读出上的SAM匹配对照

## 探索重点

保留有效Gaussian128位置/权重/NLL/BCE/RPS1，改优化器扰动更新SAM rho0.05。既有SW的SAM局部收益仅为动机，不能声称PERM已有效。两候选adamw/sam，同参数82720和同初始函数，80/14；真实新AdamW控制避免历史18与14预算混淆。

## 已探索内容与验证

准备实现完成，尚未启动两项、无分数。复用已有SAM双前后向、RNG回放、参数clone精确回滚、每组clip2；只改rho0.05，与无扰动AdamW对照。两项独立SOURCE起点，无已训154头；同样训练窗口和64更新/轮。SAM每更新两前后向、控制一前后向，等更新预算不等计算预算，分别记录单项时间及完成两项总墙钟/并发/显存/数据预热计入。固定POR151/SW104实际合并，五阶段。

## 收益

尚未证实，预定义sam−adamw同预算机制差值和对历史正PERM89.405564856919的绝对差值。未完成两项不声称最终消融收益。

## 不值得继续的点及证据

低LR163在14轮无改善，不继续LR网格；尺度/Laplace/gate/适配局部配置已关闭，结论仅限实测配置。此项无收益不继续SAM半径网格。


## 显式启动控制项

POR164 linear已完成，暂保留权重至配对比较结束；非线性仍运行。已用空槽显式启动165 AdamW控制，SAM尚未启动，无第三GPU任务或自动队列。私有骨干166已准备但尚无训练/分数，等待现有预定义对照。

V165 adamw: actual PERM=89.344760174753, TOTAL=80.591193779236; delta historicalRPS=-0.060804682166; best/stop=0/14, seconds=131.987, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 sam: actual PERM=89.352248368881, TOTAL=80.593814647181; delta historicalRPS=-0.053316488038; best/stop=10/24, seconds=397.458, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 completed matched80/14 pair: SAM-AdamW PERM=+0.007488194128. Equal updates, unequal compute; original fifth only, not fullOOF.


Completed adamw: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=131.987s, launch/report wall=140.226s, peak=786.465MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.


## 最终SAM机制收益

同80/14、82720参数、初始化、数据/评分，SAM−AdamW PERM=+0.007488194128；SAM89.352248368881，仍低于历史有效154的89.405564856919。小幅单折单seed实测机制收益，保留SAM最佳和last；不升级为SOTA/完整OOF收益，继续一项门控范围验证。SAM每更新双前后向，控制一次，等更新不等计算；完成两项总墙钟518.049秒、单项耗时/显存见timing.json。错开启动且其他探索穿插，不作纯两任务并发加速结论。



# v54 SW状态分层采样

## 探索重点与固定对照
当前SW73.680963725271、组合TOTAL80.342771761689。v53的SW到POR迁移有局部收益但绝对分数不足，不继续该随机架构扩容。本轮转到训练随机性：默认状态占较多训练点时，随机窗口批次可能给非默认段不稳定的更新；检验固定分层覆盖是否改善泛化，不预断言方差或成绩一定下降。
uniform/stratified同836921参数、同v46 SW第9轮完整网络/头起点，全参数只用SW标签训练。uniform从所有原训练锚点有放回随机抽64个；stratified按锚点SW状态各有放回抽32个。状态分组仅用原64井训练标签，验证标签不参与采样，验证方式不变。
两组共同采用窗口内有效128中心点Q加权BCE+8高斯NLL的均值，再对64个窗口取均值并除3。这个平均方式与历史“批次全部有效点统一取均值”略有不同，两组一起改，不能把相对历史改进全归因于分层。
分层组每窗口乘2*N_state/N，权重由全训练锚点状态比例固定；不再按批次重归一化。分层平均的期望与uniform窗口风险相同，改变的是抽样组成及随机性。Q50保持且独立，未改默认/非默认的目标风险比例。中心锚点状态不等于窗口所有标签同状态；周围有效训练标签均正常参与。这里是状态采样，不是v14的井风险加权。

## 数据、训练与复现边界
原9曲线206、原训练64/验证16井、固定第五144665点、原评分/解码保持。1024上下文/中心128、训练侧预处理/Q、原始256支持点/800动作+默认；输入井库无标签/井ID特征，验证井库仅用于自身推理。网络初始化来自本折既有SW模型，无外部模型或旧预测投票；来源同开发折选模的局限保留。
共同新AdamWlr3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps，seed20261010。初始函数与完整参数相同，训练样本不同是变量；不要求轨迹相同。原子last_state含当前/最佳模型、优化器、调度器、RNG、早停，仅显式恢复。最多2GPU，无自动队列/旧任务/其他折/交付覆盖。

## 已做、待做与收益判定
已实现训练侧状态池和概率校正，以及共同窗口风险；开始前尚无成绩。直接真实训练，记录状态池数量/哈希、实际批次状态数/权重、梯度、曲线和源哈希，不做额外逐试验数值预检或独立审查。
待做两组训练、最佳恢复、完整三列实际评分、局部配对差与相对当前绝对差分开、候选保留或清理、PROJECTS/README/EXPLORATIONS/atlas更新。有收益候选磁盘重建；无收益不扫32/32比例/学习率，不将本配置负结果泛化所有采样方法。含初始回退选择无改善时，不以训练后两个较差成绩的差值冒称可采用的收益。
按单项与双并发总墙钟分别记录耗时/显存，含加载/预热/输入库/启动间隔，代码与收尾另计。第五严格>80.5且明显收益再考虑其余折；完整原五折OOF>82、独立井确认、新目录交付仍未完成。


## v54 uniform显式启动
{"pid": 30956, "mode": "uniform", "started_at": "2026-09-29T20:11:57.837259+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_stratified_v54.py", "--mode", "uniform", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_stratified_v54_SW_uniform_fold5"], "automatic_queue": false}


## v54 stratified显式启动
{"pid": 52132, "mode": "stratified", "started_at": "2026-09-29T20:12:16.846094+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_stratified_v54.py", "--mode", "stratified", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_stratified_v54_SW_stratified_fold5"], "automatic_queue": false}


## v54井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后stratified-uniform=+0.001729531416。详见sw_stratified_v54_20260929/FINAL.md；无其他折或交付修改。


## v54最终结果与保留边界
uniform：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
stratified：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
分层-随机SW=+0.000000000000；当前SW=73.680963725271，实际组合TOTAL=80.342771761689。新采用=False。
分层无最终匹配收益，停止当前32/32比例/LR扫描；不把训练后两个退化成绩间的小差写成有效模型。
两组同836921参数、同v46SW完整起点/144665点预测；全骨干只用SW标签与Gaussian损失。两组共同窗口均值风险，分层32/32且按训练状态比例校正，未改变状态风险目标；原Q保持。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[165.044, 161.629]秒；双并发总墙钟186.892秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1257.47, 1257.47]，代码与最终重建另计。
已删除4个本轮负检查点，共37953164字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v54实际采样记录
{"uniform": {"recorded_epochs": 18, "last_batch_default_anchor_min": 37, "last_batch_default_anchor_max": 51, "post_update_best_epoch": 1, "post_update_best_score": 73.48472067251431, "initial_recovered": true}, "stratified": {"recorded_epochs": 18, "last_batch_default_anchor_min": 32, "last_batch_default_anchor_max": 32, "post_update_best_epoch": 4, "post_update_best_score": 73.48645020393029, "initial_recovered": true}}
这些计数是每轮最后实际批次的记录；不是额外梯度方差测量。两组均第18轮早停、恢复第0轮；不声称抽样方差下降或泛化改善。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

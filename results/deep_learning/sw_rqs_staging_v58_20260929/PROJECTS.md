# v58 新加法骨干与平滑密度的适配

## 探索重点与固定对照
当前v56加法骨干SW73.713975628494、TOTAL80.354325927816，v57并行扩容无收益。本批组合两条已保留机制：v56加法骨干与v48有局部插值收益的RQS条件密度；不继续扩容、线性CDF或分位数量/尾质量扫描。
joint/headstart同840771参数、相同完整初始状态与RQS函数。继承v56第11轮TCN/注意力/加法FF/归一化/整井FiLM以及默认概率头，原高斯密度换为34输出的33分位RQS。新密度零权重，偏置由64训练井非默认SW经验分位初始化；这是新密度起点，不是父模型的原预测。
两组唯一预定义变量是训练顺序：joint首轮全网络共同训练；headstart前5轮只更新4386参数的新密度output，body/norm/condition/gate全部固定，第6轮全解冻。两组统一AdamWlr1e-4，无分支高学习率，优化器从一开始包含全部参数，但冻结期旧参数没有梯度/权重衰减/Adam状态更新。全局cosine时钟均从第1轮开始；headstart骨干少5轮更新，这是顺序机制的一部分，不冒称核心更新量相等。
不加载旧v48 RQS输出权重或预测，不混合模型；复用其原CDF/解析密度/指数尾实现，不再重复数学公式自检。历史v48平滑对照收益并不保证和新骨干组合后有收益，本轮必须看完整实测。

## 数据、训练与复现约束
原9曲线/206、Q50、原训练64井/第五验证16井144665点/seed20261010、1024窗口/中心128保持。原标签变换、256支持点/800动作加默认值与评分保持；RQS CDF差分并归一化得到网格概率。损失仅Q加权SW默认BCE与非默认解析NLL除3，默认行不计连续NLL。井输入库无标签/井ID特征，验证库仅自身推理；来源同开发折选模局限如实保留。
共同max80/pat18/min10/b64/64steps、AdamWlr1e-4/wd.01/clip2、cosine80*64至1e-5。每模型自身第0轮与训练轮共同选最佳，采用当前三通道配方另作判断。原子last_state保存全部模型、优化器/调度/RNG/早停，恢复时按当前轮次重设冻结阶段；仅显式恢复，最多2GPU，无旧任务/其他折/交付覆盖。CUDA注意力/插值反向warn_only，不宣称重新训练逐位一致。

## 已完成、待做与不继续边界
已实现新骨干接入RQS及两阶段训练，开始前无成绩。直接训练，记录首个实际第二步与第6轮解冻后的各组梯度、活跃参数量、初始函数和源哈希；结束恢复144665点最佳，实际三列评分，有价值候选磁盘重建。旧密度公式已验证且未改，不做重复数值预检/独立审查。
待完成两项对照、同起点最终收益和相对当前绝对差、单项/总墙钟/显存、负权重清理、PROJECTS/README/EXPLORATIONS/atlas更新。若先适配无局部收益，不扫描冻结轮数或学习率；若组合整体低于当前模型，不将新密度初始到训练的变化当新冠军。未完成比较不写最终收益。
第五严格>80.5且明显收益后才考虑其他折；完整原五折OOF>82、独立井确认和新目录交付尚未完成。耗时包含加载/输入库/预热/启动间隔，代码和最终重建另计。实际明显无望可提前结束。


## v58 joint显式启动
{"pid": 41876, "mode": "joint", "started_at": "2026-09-29T20:57:45.424124+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_rqs_staging_v58.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_rqs_staging_v58_SW_joint_fold5"], "automatic_queue": false}


## v58 headstart显式启动
{"pid": 50924, "mode": "headstart", "started_at": "2026-09-29T20:58:03.405608+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_rqs_staging_v58.py", "--mode", "headstart", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_rqs_staging_v58_SW_headstart_fold5"], "automatic_queue": false}


## v58井条件对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后headstart-joint=+0.120651423172。详见sw_rqs_staging_v58_20260929/FINAL.md；无其他折或交付修改。


## v58最终结果与保留边界
joint：SW 73.587566682801，最佳第4轮，停止第22轮（patience18）。
headstart：SW 73.708218105973，最佳第2轮，停止第20轮（patience18）。
先适配-共同SW=+0.120651423172；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
先适配有匹配局部收益，保留其最佳及完整状态；不将其低于现有SW的绝对成绩写成新冠军。该收益来自反复使用的开发折，未完成独立井确认。
两组同840771参数及初始RQS函数，v56加法骨干和默认头保留；新密度训练侧先验。headstart前5轮仅4386密度参数，第6轮全解冻，其余相同；全程只用SW标签及原Q/RQS解析损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=True。不重复未改密度公式检查。
单项耗时分别[222.114, 191.98]秒；双并发总墙钟226.230秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1277.44, 1278.72]，代码与最终重建另计。
已删除2个本轮负检查点，共19070070字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v58真实阶段与绝对边界
{"joint": {"stopped_epoch": 22, "best_epoch": 4, "density_only_epochs": 0, "backbone_update_steps": 1408, "selected_body_parameter_change_L2": 1.4109308771569136}, "headstart": {"stopped_epoch": 20, "best_epoch": 2, "density_only_epochs": 5, "backbone_update_steps": 960, "selected_body_parameter_change_L2": 0.0}}
先适配最佳第2轮发生在仅密度更新阶段，选定骨干变化L2为0；第6轮确实解冻，所有组梯度非零，但之后未超过第2轮。不能把此次收益解释为解冻后全网络提升。两组初始SW72.415319274399相同，先适配最终73.708218105973较共同73.587566682801高0.120651423172，但比当前73.713975628494低0.005757522521。先适配实际替换TOTAL80.352310794934，当前组合80.354325927816保持。
保留先适配best/last与完整144665点磁盘重建凭据；共同训练负权重已删除，原RQS/v56及其他正候选、交付保留。分位点/尾质量/公式均未改，旧数值验证不重复。复用的通用配置描述提到linear/RQS是实现支持范围，本批density_mode固定rqs，两项只差阶段；未训练linear。独立井确认、第五严格>80.5及完整OOF>82仍未完成。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

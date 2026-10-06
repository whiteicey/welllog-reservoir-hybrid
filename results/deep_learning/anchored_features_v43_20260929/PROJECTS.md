# v43 SW完整特征骨干：原权重约束迁移

## 探索重点与预定义对照
从当前TOTAL80.336513436816、SW73.663082797066开始。v42原点约束相对free训练后最高+0.002969806658，18/18对应轮次更高，但都低于原模型。保留这个减少退化的局部机制，检验它是否允许更充分的特征学习。
本批仅free/anchored两项。两个私有专家中除main_head/aux头之外的所有既有特征参数可训练，包含输入校准、局部/全局表征、编码/解码和逐点融合分支；原目标输出头、内部输入路由、v26/v28/v29修正权重保持冻结，计算对骨干仍可微。无新增容量，不冷启动，不读其他目标标签。
两组同初始化/可训练参数/训练窗口/优化器/评分，唯一差别仍为原点惩罚系数0/1：各参数张量 mean((p-p0)^2)/max(mean(p0^2),1e-4) 的平均。p0来自原训练折已训练模型，无伪标签或验证数据监督。
相对v42，更新范围更大且使用已有私有骨干微调的保守lr1e-5；不是宣称两批只有一个因素不同。因果对照仅本批free与anchored。v42只末端dec0的当前系数/LR扫描仍关闭。

## 固定条件
原9曲线206主特征与既有63幅值侧特征、64训练井/16原第五验证井、Q50、SW NLL/3、原解码/评分和seed20261010保持。80最大轮/patience18/min10/b64/64步；AdamW lr1e-5/cosine80*64至1e-6、wd0、clip2。降低整个调度比例，周期仍80轮。所有dropout保持原eval状态，不修改验证统计。
每项独立优化器、早停和最佳checkpoint，保留第0轮回退；原子last_state含参数/优化器/调度器/RNG/停止状态。每次恢复须显式启动。最多2GPU任务，显式启动两项，无自动队列。

## 已做、收益、计划与未做
已实现新模块和训练器；本批收益尚未证实。真实训练记录梯度范围与冻结状态，结束恢复最佳预测并固定POR/PERM作实际三列评分，正候选另做磁盘重建。总墙钟与单项耗时/显存分开记录，加载/预热/处理计入。
若无新增或匹配消融收益，不继续扫描本完整骨干的约束系数/LR。旧v8端到端训练同时更新输出头/路由且起点不同，本批不冒充完全复跑。
完整原五折OOF>82目标未完成；第五折须严格>80.5且有明显收益才考虑其他折。本批无其他折、旧队列恢复或交付覆盖。维护PROJECTS.md的目标/已做/计划/未做/后续、README、EXPLORATIONS与图谱。目标完成后可选新井独立验证与新目录交付，当前未做。


## v43 free显式启动
{"pid": 57296, "mode": "free", "started_at": "2026-09-29T16:06:07.224774+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_features_v43.py", "--mode", "free", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_features_v43_SW_free_fold5"], "automatic_queue": false}


## v43 anchored显式启动
{"pid": 68220, "mode": "anchored", "started_at": "2026-09-29T16:06:23.191820+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_features_v43.py", "--mode", "anchored", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_features_v43_SW_anchored_fold5"], "automatic_queue": false}


## v43完整特征骨干对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见anchored_features_v43_20260929/FINAL.md。

## v43归档结论
两项18轮结束，训练后SW最高free73.505103279、anchored73.507620278，均低于原73.663082797；实际三列TOTAL80.336513436816保持。约束相对+0.002516999785，11/18对应轮次更高，中位+0.001984832480，完整骨干未证实稳定收益。关闭本配置LR/系数扫描，后续需切换机制或目标，尚无新任务。删除本批4个未采用权重共59620738字节，原冠军/v42局部对照/交付保留，源/配置/轨迹/紧凑预测保留。清单见deletion_manifest.json。第五折>80.5与完整OOF>82未完成，无其他折、活动训练或队列。



## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

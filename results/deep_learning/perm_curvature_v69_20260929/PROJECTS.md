# v69 PERM分位间距曲率约束

## 探索重点与固定对照
从v67正候选RQS第24轮PERM88.827250出发；v68冲突投影无最终收益，不继续其阈值扫描。新假设是分位间距过度起伏可能追逐训练样本，固定增加间距平滑惩罚。仅一项curvature，复用v68 sum已完成对照（含起点88.827250、训练后最高88.807819），不重训未改控制。
相同838211参数、完整函数、初始张量、优化器和两次autograd。惩罚为相邻32个正分位间距log后的二阶差分平方均值，只对训练非default PERM点有效，按原Q加权并对全部监督点平均；总目标原BCE/NLL+0.05*惩罚/3。仍按sum直接相加门控与（密度+惩罚）梯度，不做投影。此项只约束分位间距起伏，不是对物理密度曲率的精确积分，也不强制密度单峰。

## 数据训练边界和已做
原9曲线/206输入、64训练井拟合/Q50、第五16井144665点、seed20261010、1024窗口中心128、原RQS公式/解码/评分不变，仅PERM标签训练；整井输入库无标签/井ID，验证井只供自身推理。全网络AdamW3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。历史控制不是同步双跑或逐位重训保证，比较初始状态/预测及配置相同。
已实现，未训练/无成绩；直接运行实际梯度、来源哈希、初始/最佳144665点和三列评分核验，正候选保留前磁盘重建。无独立预检/审查。原子完整last_state保存网络/优化器/调度/RNG/停止，只显式恢复；原v67模型保留。

## 收益、待做与不继续
待新候选训练、与历史控制配对及当前绝对差、保留/清理/PROJECTS/EXPLORATIONS/图谱。预固定系数0.05；若无最终配对或源模型增益，不继续系数/差分阶数扫描，只限当前PERM间距约束，不泛化所有密度正则化。初始回退不算训练收益。
本轮一个新任务，历史对照耗时单列；计入加载/预热/训练/评估，代码/收尾另计，不把历史并发墙钟算作本轮双跑。最多2GPU、无队列/其他折/旧交付修改。负模型先清单后删除。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82仍未完成。目标后独立确认和新目录交付。


## v69 curvature显式启动
{"pid": 49620, "mode": "curvature", "started_at": "2026-09-29T23:01:38.292403+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_curvature_v69.py", "--mode", "curvature", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_curvature_v69_PERM_curvature_fold5"], "automatic_queue": false}


## v69最终结果
一个新候选完成，复用历史sum控制，TOTAL=80.354325927816；curvature-sum PERM=+0.000000000000，训练后差=+0.011725166945；采用=None，保留=[]。负权重清理，源/交付不改，无其他折；目标未完成，见perm_curvature_v69_20260929/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

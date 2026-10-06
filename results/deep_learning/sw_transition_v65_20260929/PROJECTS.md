# v65 SW相邻状态监督

## 探索重点与预定义对照
从当前v56 additive SW73.713975628494、组合80.354325927816出发，不继续POR容差损失或内部适配器扫描。固定detached/joint两项，同一完整SW骨干和初始函数，新增相同8257参数训练辅助头（256→32→1）。输入为相邻归一化隐藏特征的均值和绝对差，标签为两点SW default状态是否不同。辅助头不参与推理，绝不把真实状态作为模型输入。
仅改变辅助损失是否反传到骨干；detached组同样训练辅助头但detach特征，joint组联合反传。原SW主损失相同；辅助0.1*平衡BCE，正/负类各自Q加权损失均值后平均/3。只取中心128内两端均有标签且同井相邻原行的点对；井界、无效/未标注、重复填充不参与。标签仅原64训练井。

## 已做与训练边界
原9曲线/206输入、训练井拟合预处理、输入整井库不读标签/ID、原第五16井144665点、Q50、seed20261010、1024随机窗口中心128、原Gaussian+BCE/评分/解码保持。全部骨干lr3e-5，新辅助头lr3e-4；两参数组独立clip2以免detach对照因辅助梯度影响主梯度裁剪。AdamW wd.01、cosine80*64至1e-5、max80/pat18/min10/bs64/64steps。完整初始状态相同；将原函数纳入最佳选择，回退不算学习收益。
已实现，尚未训练/无成绩；实际训练中记录辅助到特征的梯度joint>0/detached=0。完整原子last_state包含网络/优化器/调度/RNG/停止状态，仅显式恢复。先串行，最近双骨干并发OOM后不把max2视作默认并发；本轮不需要双骨干，但保守控制总显存。

## 待做、收益和不继续
直接运行两项，比较最终选定SW与训练后最高SW，实际固定POR/PERM后三列评分。保留正候选前磁盘原始输入重建，源哈希/初始预测/最佳恢复作为结果证据，不增加独立预检或审查。若joint无最终配对收益，停止当前状态差辅助系数/头宽度扫描；结论只限此目标，不泛化全部辅助学习。
分别记录单项耗时与串行总墙钟，含加载/预热/训练/评估，代码和收尾另计。负权重清理前写删除清单。第五严格>80.5且明确收益后才考虑补折；完整原五折OOF>82未完成，无旧队列/其他折/交付覆盖。维护PROJECTS、EXPLORATIONS和图谱；目标完成后独立确认及新目录交付。


## v65 detached显式启动
{"pid": 61196, "mode": "detached", "started_at": "2026-09-29T22:24:40.868917+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_transition_v65.py", "--mode", "detached", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_transition_v65_SW_detached_fold5"], "automatic_queue": false}


## v65 joint显式启动
{"pid": 57156, "mode": "joint", "started_at": "2026-09-29T22:26:46.861211+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_transition_v65.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_transition_v65_SW_joint_fold5"], "automatic_queue": false}


## v65最终结果
两项串行完成，TOTAL=80.354325927816；joint-detached SW=+0.000000000000，训练后差=-0.017791342092；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_transition_v65_20260929/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

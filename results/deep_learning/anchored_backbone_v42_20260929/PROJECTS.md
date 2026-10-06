# v42 SW末端残差块与原权重约束

## 探索重点、假设及对照
从原第五折TOTAL80.336513436816、SW73.663082797066出发。冻结头小改多次无收益，v39仅norm和v40井不变性也无收益；本次允许已有末端残差块卷积修改局部表征，并检验参数原点约束能否减少微调退化。不是网络扩容或冷启动，不延伸分布族小头。
free与anchored均只训练两个私有专家的dec0全部原参数，其余骨干/头/路由/幅值小头固定。前向仍对这些残差块可微，eval/dropout状态保持。相同可训练参数、初始化、窗口、优化器；唯一差别为原权重约束系数0或1。
约束=各参数张量 mean((p-p0)^2)/max(mean(p0^2),1e-4) 的平均，p0为本训练折已训练原模型权重。不是伪标签，不读验证标签产生监督；无原预测蒸馏。对每个张量做RMS归一化，固定0.01 RMS下限，初始惩罚为0。
历史重叠窗口一致性对两个偏移窗口同时反传、第五折80.1022不达标；本次不重复该机制。v39使用norm而本次使用完整dec0卷积块，两个实验不能视为仅一个因素的对照；本次free/anchored才是匹配消融。

## 训练与选择
原64训练井/16验证井、9曲线206主特征及原63幅值侧特征、Q50、SW NLL/3、原解码和评分不变。80最大轮/pat18/min10/b64/64步；AdamW lr1e-4/cosine80*64至1e-5、wd0、clip2，两组相同。每组独立优化器/早停/按官方SW选最佳；保留第0轮回退。保存原子完整训练状态，恢复需显式启动。
最多2GPU进程，无自动队列。分别记录单项耗时、峰值显存和相同工作量总墙钟。原POR/PERM保持独立且不训练；最后固定3模型实际组合评分。

## 已做、收益及待做
已实现模块/训练器，收益尚未证实；启动后记录真实快照，结束记录更新后最高和含原模型回退的分数，不能将第0轮回退视作训练收益。
若两项均无收益，不继续本系数/学习率网格扫描；结论只限当前SW起点/更新块与固定配置。
目标仍为第五折严格>80.5、有明确收益后再考虑其余折，完整原五折OOF>82未完成。本批无其他折、旧队列重启或交付覆盖。维护PROJECTS.md、独立README、EXPLORATIONS和图谱。目标达成后的可选工作为新井独立验证与交付，当前不启动。


## v42 free显式启动
{"pid": 61292, "mode": "free", "started_at": "2026-09-29T15:54:48.717088+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_backbone_v42.py", "--mode", "free", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_backbone_v42_SW_free_fold5"], "automatic_queue": false}


## v42 anchored显式启动
{"pid": 61076, "mode": "anchored", "started_at": "2026-09-29T15:55:09.156414+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_backbone_v42.py", "--mode", "anchored", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_backbone_v42_SW_anchored_fold5"], "automatic_queue": false}


## v42末端块原权重对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见anchored_backbone_v42_20260929/FINAL.md。

## v42归档与下一步边界
两项均18轮完成，固定组合TOTAL80.336513436816。原点约束相对free训练后最高SW+0.002969806658，18/18匹配轮次更高，中位差+0.108804776891；仅是单种子开发折内减少退化的证据，未超原SW。删除负free状态与两份第0轮重复模型共3个、4366424字节；保留anchored完整last_state约4.2MB用于局部正机制对照，不作为新交付。清单见deletion_manifest.json。无活动训练或队列；原其他折未启动。后续可在不同更新部位或目标的独立假设中携带原点约束，尚未实现或启动；不继续本dec0的系数/LR扫描。



## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

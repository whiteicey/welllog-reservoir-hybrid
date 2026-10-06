# v78 SW跨注意力头失活掩码对照

## 探索重点与相同条件
当前v77 p40独立通道组合TOTAL80.363582620846，SW73.740423322864。40%失活在16井8胜8负，较20%数值子群+0.286177但默认-0.111979；绝对收益很小。本批从完整p40第25轮函数出发，比较shared/per_head两项：4层各四头共用随机key掩码，或每头独立掩码。假设独立头保留更丰富的联合上下文覆盖，减少单次全头遗漏；可能增加训练噪声，收益未知。
839481参数、初始完整状态与预测相同；仅改变同BHt随机张量是否广播头0，每层两组消耗相同BHt随机数。每头全丢回退原valid、padding保持、掩码对查询共享。验证无失活、一次完整前向，未新增融合或改变解码。不是旧1e-4轨迹续跑：两组均新AdamW3e-5/wd.01/clip2，cos80*64至1e-5，优化器重置。旧v77不能直接作为新随机序列的同训练控制，所以本批shared也重训。
原9曲线206、64训练井/第五16井144665点/Q50/seed20261010、1024随机上下文中心128、SW自身BCE+8Gaussian NLL、256支持/800动作、80/18/min10/b64/64steps不改。全部SW参数更新，POR/PERM固定；只用训练标签，输入侧井上下文不含井ID或标签。完整原子状态/RNG，仅显式恢复。

## 已探索、收益与不继续
已实现；启动时尚无新成绩。直接训练，初始完整预测检查属于实际训练评估，不单独预检或启动审查。记录训练真实梯度、最优恢复、最高训练后得分和含起点回退的分数，结束实际组合验证。分别报告per_head-shared、各自相对完整起点、默认/数值子群及井差；开发折反复选择不当独立泛化证明。
per_head若无最终配对收益，关闭当前头间掩码拆分的概率扫描；若shared仍有绝对收益可独立保留，不把共同低学习率微调收益归给独立掩码。只保存有绝对/明确配对收益的候选；负checkpoint先写清单后删除，来源与旧交付保留。最多2GPU，实测资源后显式启动第二项，无自动队列；记录单项及同工作量总墙钟、并发/显存，加载预热训练验证保存与启动间隔计入，代码/收尾分开。
未启动旧任务或其他折。固定第五严格>80.5且明确收益才考虑补其余折；完整原五折OOF>82仍未达成。维护PROJECTS/README/EXPLORATIONS/图谱；目标完成后独立井确认与新目录交付。


## v78 shared显式启动
{"pid": 63868, "mode": "shared", "started_at": "2026-09-30T09:23:18.042260+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_heads_v78.py", "--mode", "shared", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_heads_v78_SW_shared_fold5"], "automatic_queue": false}


## v78 per_head显式启动
{"pid": 31940, "mode": "per_head", "started_at": "2026-09-30T09:23:52.214704+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_heads_v78.py", "--mode", "per_head", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_heads_v78_SW_per_head_fold5"], "automatic_queue": false}


## v78最终结果
两项完成，当前独立通道路线TOTAL=80.363582620846，相对起点+0.000000000000；per_head-shared SW=+0.000000000000，训练后差-0.024271478516。采用=None，保留=[]；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_dropout_heads_v78_20260930/FINAL.md。


## v78关闭范围与实际结论
两项都恢复原v77第25轮完整函数（本批最佳轮0），不冒称新训练候选提升。per_head训练后最高较shared低0.024271478516，含起点两组差0；当前头间独立掩码路线关闭，不继续其概率细扫，也不重复相同低学习率全网络微调。结论限于这次完整p40热启动/3e-5/4层掩码，不能推出所有独立头正则无效。当前研究配方TOTAL80.363582620846保持；四个负/重复checkpoint按清单删除，原v77及来源模型保留。
无新正候选，不额外重复原始输入GPU重建；实际训练已核对初始全点等于父模型、最佳恢复与三列评分。仍未过严格80.5，无其他折，完整OOF>82目标未完成。后续换输入表征或输出学习机制；未准备自动队列。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

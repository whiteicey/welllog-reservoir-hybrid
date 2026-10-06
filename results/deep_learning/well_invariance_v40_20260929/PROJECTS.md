# v40 训练井不变性：阻断与反转梯度

## 探索重点与历史区别
固定80.336513436816基线，针对v39单纯SW归一化微调无收益，检验训练井特征依赖是否妨碍泛化。旧well_adversarial_sequence是多目标从头训练、单128维线性分类器、系数至.01，原第五79.843176；本次是强SW双专家热启动、仅5120个归一化仿射参数更新，配备同一个256→64→64训练井分类器。不是首次尝试井对抗思想，也不重复旧任务。

## 预定义匹配对照
blocked/reversed两个独立进程，同完整初始化、同NLL/Q50/采样窗口/优化器/参数集合。两组都训练井分类器；blocked井分类梯度不进骨干，reversed在前三轮也阻断，第4至8轮将反转系数从.2升到1，之后固定1。总损失SW原NLL/3 + .1*井分类CE。分类器正常最小化CE，骨干在reversed中收到相反方向的井分类梯度。
两个专家主读出前128维隐藏特征，各自按有效中心128点平均，拼成256维；训练井ID仅用作分类监督，不作为预测输入。只映射64训练井，16验证井的域标签固定为-1，训练锚点严格检查有效映射。验证不会运行或更新分类器。无验证标签/井参与辅助拟合、无跨井输入窗口。
原5120个GroupNorm/LayerNorm缩放偏置更新，旧卷积/头/路由固定。新增20608参数仅属于训练井分类器，推理不用其输出；两个候选总可训练参数相同。真实梯度和归一化权重变化记录，避免把分类器单独学会当作骨干有效训练。

## 训练与验证边界
原9曲线/206输入与63既有幅值侧特征、原第五折144665点评分/动作不改。max80/pat18/min10/b64/64步；norm AdamWlr1e-4/wd0，分类器lr3e-4/wd.01；cosine80*64至1e-5，两个参数组各clip2。两进程并发上限2，无自动队列/恢复。
初始模型等于原SW73.663082797066；独立按SW选最佳，完成后恢复并和固定POR/PERM实际组装。不做重复预检/独立审查。训练侧分类正确率仅作诊断，需与训练井最大先验类比例比较，不作为预测评分。

## 已完成、未完成与后续
已实现两候选；训练收益待定。若井分类器不能学习，本次只能说明该训练设置信号不足；若能学习但反转无收益，则不继续该对抗系数扫描。不将训练井可区分性本身当作泛化失败的因果证据。
完整OOF>82尚未达成，第五严格>80.5并有明显收益才考虑其余折；本批无其他折/旧任务/交付覆盖。每批维护PROJECTS.md、README、EXPLORATIONS和图谱，保留真实失败与负结果，删除无用checkpoint。


## v40 blocked 显式启动
{"pid": 66276, "mode": "blocked", "started_at": "2026-09-29T15:17:41.762513+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_invariance_v40.py", "--mode", "blocked", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_invariance_v40_SW_blocked_fold5"], "automatic_queue": false}


## v40 reversed 显式启动
{"pid": 22660, "mode": "reversed", "started_at": "2026-09-29T15:18:15.197884+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_invariance_v40.py", "--mode", "reversed", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_invariance_v40_SW_reversed_fold5"], "automatic_queue": false}


## v40骨干归一化对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见well_invariance_v40_20260929/FINAL.md。


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 before reversal starts. For the prescribed active epochs4..18, blocked best73.5954025454393 and reversed best73.59465943798016 (delta-0.000743107459). Final SW difference-0.006539743169. This diagnostic does not change checkpoint selection. Final domain accuracy blocked0.165039/reversed0.162842 vs majority prior0.022486: classifier learns, but the reversed objective does not improve this SW adaptation setting.
Do not extend this adversarial coefficient/curriculum scan. Next pending direction: inspect PERM conditional-density family/loss alternatives against the current retained PERM89.396724 model; preserve original actions/scoring and distinguish an initial distribution intervention from a retraining benefit. No new PERM run implemented or launched yet. Original fullOOF>82 and fifth>80.5 remain outstanding.


## v40 retention and terminal state
Both candidates finished with no adopted model. Deleted4 redundant checkpoints totaling9012676 bytes after verified restoration and fixed3column scoring; SHA256/size manifest saved. Source/config/log/compact predictions/diagnostics retained. No live training or queued launch, no other folds and no delivery changes. Current recipe still TOTAL80.336513436816.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

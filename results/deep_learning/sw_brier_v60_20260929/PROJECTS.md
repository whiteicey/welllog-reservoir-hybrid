# v60 SW概率平方损失对照

## 探索重点
从已保留v58 headstart第2轮SW73.708218开始；当前组合SW73.713976/TOTAL80.354326。v59读出扩展无最终收益，不再扫描。此次无新增模块，冻结body/norm/FiLM，只训练原gate和RQS密度，共4515参数（总840771）。两组参数、起点和样本相同，只差辅助项系数。
likelihood使用原Q加权BCE+连续NLL；brier额外加入0.5*(gate概率平方误差+16*非default条件256格Brier)。Brier为sum(p²)-2*p_true+1，与容差覆盖、分位pinball不同；类别按原归一化density_edges由训练SW标签确定，开放外侧尾部覆盖所有值。因子16固定，不扫描。它是离散概率训练目标，不声称等同官方效用或连续CRPS；保留原连续NLL。

## 数据与执行约束
原9曲线/206输入、64训练井/16验证井/原seed20261010、Q50及原解码不变。缓存冻结骨干1024上下文的中心128点，训练与验证分包；canonical固定训练上下文是两组共同变化，不将相对旧随机锚点绝对差全部归因于Brier。只读取训练包SW标签作损失；验证只选官方SW最佳。原完整起点预测及人口覆盖在缓存运行时检查，保留候选需原始输入144665点重新推理，不增加数学预检或独立审查。
新AdamW1e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。门控和密度同训练；原子保存best及last完整optimizer/scheduler/RNG/早停状态，仅显式续跑。最大2GPU，无其他折/旧队列/交付覆盖。

## 已做、收益、未做与关闭条件
已实现匹配目标，尚未训练，收益尚未证实。待共享缓存、两项训练、配对差与当前绝对分、原始输入保留模型验证、清理负权重和缓存，更新PROJECTS/EXPLORATIONS/atlas。
若Brier无最终配对收益，关闭此256格平方损失系数扫描，不泛化所有概率评分无效。第0轮回退不算学习收益。开发折重复使用，不能写成独立新井收益；完整五折OOF>82、第五严格>80.5、独立确认与新目录交付未完成。
共享缓存准备与单项训练、双并发总墙钟分别记录；含加载/预热/评估，代码间隔如实单列，不把并发耗时相加当墙钟。目标达成后可独立确认及新目录交付。


## 共享缓存完成
实际范围、时间、源函数核验见cache_manifest.json；两个候选尚未训练。


## v60 likelihood显式启动
{"pid": 30712, "mode": "likelihood", "started_at": "2026-09-29T21:30:13.039963+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_brier_v60.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_brier_v60_SW_likelihood_fold5"], "automatic_queue": false}


## v60 brier显式启动
{"pid": 68940, "mode": "brier", "started_at": "2026-09-29T21:30:17.826971+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_brier_v60.py", "--mode", "brier", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_brier_v60_SW_brier_fold5"], "automatic_queue": false}


## v60最终结果
实际TOTAL=80.354325927816，Brier-likelihood SW=+0.000000000000，采用None，保留[]。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，耗时如实记录。详见sw_brier_v60_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

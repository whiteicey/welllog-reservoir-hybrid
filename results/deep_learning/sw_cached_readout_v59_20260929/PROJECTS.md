# v59 完整RQS密度读出：仿射与非线性

## 探索重点和固定对照
v58先适配头有同起点+0.120651SW收益，最佳在仅密度第2轮；SW73.708218略低当前73.713976。本轮沿有效的新密度读出研究非线性表达，不继续骨干扩容或冻结时长扫描。
从v58 headstart第2轮完整模型出发，body/norm/FiLM/default gate全部固定。保留34输出原密度线性核并可训练，再加入128→64→34零输出分支；affine中间为恒等，silu中间为SiLU。两组851237总参数、14852可训练密度参数、初始参数和完整函数相同；新增最后层权重/偏置为零。仿射组是线性映射的过参数化，非线性组改变函数表达能力，不能把参数量相同误解为函数族相同。输出仍是完整RQS密度，默认头固定，不做旧预测投票。

## 缓存、数据与训练边界
共享缓存只保存固定骨干的输入推理结果，训练64井与验证16井分包。按原推理计划每128点一个中心块，骨干读1024上下文，保存中心128点归一化隐藏向量及冻结默认logit。训练采用固定canonical窗口，替代历史随机锚点上下文；这是两组共同变化，历史绝对差不能全归因于激活。对照仍原9曲线206、Q50、原井集合和144665验证点、原目标变换/256支持点/800动作及评分。
仅训练包的SW标签进入损失。缓存构建核对训练raw集合、验证覆盖和井隔离，并用完整144665点初始预测核对源模型；未使用验证标签拟合特征。验证密度头按原B×1024点行布局运行，外围填零不影响逐点头，避免改变矩阵形状造成初始数值差。候选保留前再做原始输入完整骨干推理验证，缓存分数不能替代它。
新AdamWlr1e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64个缓存中心块/64steps，seed20261010。全部密度参数共同训练，按官方SW选择最佳；原子last_state含头/模型、优化器/调度/RNG/早停，恢复须显式。最多2GPU，无旧队列/其他折/交付覆盖。

## 已做、待做、收益与关闭条件
已实现两种读出和共享缓存构建，尚无候选训练成绩。直接运行缓存与训练，运行中检查真实数据覆盖和冻结状态，不加重复数学预检/独立审查；RQS公式与原评分未改。待完成两项训练、同参数对照/绝对分数、保留模型原始输入重建、负检查点及可再生缓存清理、PROJECTS/README/EXPLORATIONS/atlas。
若SiLU无最终配对收益，不扫描隐藏宽度/激活函数；只限该固定骨干及RQS参数化，不否定其他非线性头。若有局部收益但低当前模型，可保留机制候选。第五严格>80.5且明显收益才考虑其他折；完整原五折OOF>82、独立井确认和新目录交付未完成。
共享缓存只构建一次，准备耗时与单项训练、两任务总墙钟分开，完整流程从准备到训练结束的实际墙钟另列；如有编写间隔如实区分，不把两项并发耗时相加冒充总墙钟。显存/预热/读取均记录，代码与最终重建单列。缓存可由保留源模型和数据重建，不作为新测试数据或精度证据。


## 共享缓存完成
实际范围、时间、源函数核验见cache_manifest.json；两个候选尚未训练。


## v59 affine显式启动
{"pid": 48032, "mode": "affine", "started_at": "2026-09-29T21:22:54.685217+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_cached_readout_v59.py", "--mode", "affine", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_cached_readout_v59_SW_affine_fold5"], "automatic_queue": false}


## v59 silu显式启动
{"pid": 73624, "mode": "silu", "started_at": "2026-09-29T21:22:59.543433+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_cached_readout_v59.py", "--mode", "silu", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_cached_readout_v59_SW_silu_fold5"], "automatic_queue": false}


## v59最终结果
实际TOTAL=80.354325927816，SiLU-affine SW=+0.000000000000，采用None，保留[]。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，失败和耗时如实记录。详见sw_cached_readout_v59_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

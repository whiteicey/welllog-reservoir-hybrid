# v52 PERM内容自适应深度汇聚

## 探索重点与固定对照
v51训练增强无最终收益，当前PERM89.396724/TOTAL80.342772保持。新假设：原骨干每16深度点平均为一个注意力token，可能弱化局部薄层信号；用内容相关汇聚检验这一结构限制，不继续增强概率/Beta/层扫描。
position/content两个完整私有PERM网络，同837305参数、同源SW第9轮骨干/FiLM、同完全重置的PERM25维输出。源与训练预算沿用v50已证实局部正收益的迁移配置，不从已过拟合PERM最后状态继续微调。增加96→4无偏置权重打分器，四组各24通道值，在每16有效深度点内softmax加权汇聚，再送原四层长程注意力。初始打分全零，对应各组有效点均值。
两组只改打分器输入：content读三层局部TCN后的96维测井内容；position读GELU(原位置编码(relative depth, window position))。四组值、下游结构、参数量、初始化、标签、优化器相同；位置输入也可学习非线性位置偏好。名义容量相同，但输入连通/有效表达能力不同是预设实验变量，不能单独归因于参数数目。空token权重与输出为零，有效掩码内归一。
全部骨干/FiLM/位置/打分器/输出同时训练，不是冻结模型的输出小补丁。旧v50等权汇聚作为历史参照，不重新跑同一对照；本批position/content是严格配对。与旧预测概率池化/多窗口投票不同，本次仅内部特征压缩，不混合旧模型输出。

## 数据与训练边界
源权重仅原64训练井SW监督训练；当前PERM两组输出全重置，损失仅PERM训练标签。原9曲线206、Q50、固定原第五144665点/seed20261010、1024窗口/中心128、原8Gaussian/默认头与256支持点/800动作加默认保持。输入井库无标签/井ID特征，验证井库只用于该井推理；无跨井增强。
同AdamWlr3e-4/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64步。输出训练侧先验相同，不把先验到88分的学习当超越当前模型。完整原子last_state、显式恢复，最多2GPU无自动队列；实际明显落后可提前停止。CUDA注意力/插值反向warn_only，不声称重新训练逐位一致。

## 已做、待做与不继续边界
已实现四组掩码汇聚、两种打分输入及训练器；开始前无新成绩。直接真实训练，记录打分器梯度、权重范数及训练池化熵，区分实际学习与失活。结束核验同初始状态/先验预测、源哈希、最佳恢复和实际三列评分；有价值候选磁盘重建。不重复原密度检查。
待做两项完整对照、与同容量/历史v50/当前最优三种差值分开、负权重清理、PROJECTS/README/EXPLORATIONS/atlas更新。单项和双并发总墙钟/显存分开，含加载/预热/输入库/启动间隔；代码与最终重载另计。
若内容汇聚不优于位置对照，不扫描汇聚组数、步长或温度；只限此结构和训练配置，不能否定所有自适应token化。原第五严格>80.5且明确收益后才考虑其他折；完整OOF>82未完成。无旧任务/其他折/交付覆盖；独立井确认及新目录交付尚未开始。


## v52 position显式启动
{"pid": 44772, "mode": "position", "started_at": "2026-09-29T19:28:20.558046+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_token_pool_v52.py", "--mode", "position", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_token_pool_v52_PERM_position_fold5"], "automatic_queue": false}


## v52 content显式启动
{"pid": 61280, "mode": "content", "started_at": "2026-09-29T19:28:39.215019+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_token_pool_v52.py", "--mode", "content", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_token_pool_v52_PERM_content_fold5"], "automatic_queue": false}


## v52井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后content-position=-0.026171081763。详见perm_token_pool_v52_20260929/FINAL.md；无其他折或交付修改。

## v52最终结果与清理
position第22轮最好PERM88.894217488422，第40轮patience18结束；content第24轮最好88.868046406659，第42轮结束。内容-位置=-0.026171081763，无匹配收益。两者也低于历史v50普通平均汇聚89.050807574234（分别-0.156590085813/-0.182761167576），以及当前PERM89.396724049366。历史比较与本批同容量比较分开，不能把初始先验68.383772到88分的学习当超越当前模型。
两项初始状态/输出头/144665点先验预测哈希一致；真实打分器梯度非零，选定权重范数position0.253729/content0.350035，内容权重熵训练中已偏离均匀值，非失活试验。最大池化权重可受边界仅少数有效点影响，不将单个max数值当全窗口尖锐性证据。
最佳恢复全144665点一致、来源哈希与真实三列组合评分完成。没有采用新模型，CURRENT_RECIPE仍TOTAL80.342771761689；严格第五>80.5及完整OOF>82未完成，无其他折。负候选不追加重复磁盘/原密度验证。
单项379.7283/389.8115秒，总墙钟414.115675秒，max2GPU，峰值1319.99/1291.71MiB；含加载/预热/输入库/启动间隔，代码与最终收尾另计。全部结束无训练队列；删除四个本轮best/last负权重，保留代码/配置/日志/预测/梯度与紧凑报告。v50迁移、v48样条、当前三目标及源依赖/旧交付不改，哈希/字节见deletion_manifest.json。
关闭当前汇聚组数/步长/温度扫描，不否定所有自适应token化。下一批需换机制，尚未实施，不能算作收益。独立井确认、完整五折及新目录交付仍未完成。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

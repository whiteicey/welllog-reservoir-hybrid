# v67 PERM分位密度

## 探索重点与预定义对照
将既有SW平滑CDF的局部消融机制迁移至PERM；原v50 sw_transfer已保留，PERM89.050807574234，低于当前双专家89.396724049366。本轮linear/rqs两个私有PERM候选从同v50训练折来源加载骨干/FiLM/默认门控，只重置连续密度。新838211参数模型预测33个有序分位点，两端指数尾各质量.005。linear分段线性CDF，rqs有理二次CDF，斜率来自相邻割线调和均值；沿用v48已训练源码，不重复公式预检。
相同参数张量与分位先验、优化配置；插值函数不同，初始CDF/预测可能不同，不能冒称初始完整函数一致。密度先验只用64训练井非default PERM标签，保留原归一化目标；原父密度函数不保留。比较最终配对差和相对当前冠军绝对差，单独报告初始差，收益不由起点变化冒充。

## 数据训练边界与已做
原9曲线/206输入、训练侧预处理/Q50、第五16井144665点、seed20261010、1024随机窗口中心128、原256支持点/800动作+default官方评分不变。整井128输入库无标签/井ID，验证井仅自身推理。训练仅PERM BCE与正确新CDF导数负对数，所有骨干/门控/密度/FiLM可训练，POR/SW固定不更新。
统一AdamW1e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps；完整原子last_state含模型/优化器/调度/RNG/早停，仅显式恢复。已实现，启动时无成绩。运行实际梯度、来源哈希、同初始张量、全144665最佳恢复及三列评分核验；正候选保留前磁盘原始输入重建，不做独立数学预检/审查。

## 收益、待做与不继续
待两项训练及完整配对消融、当前绝对差、负模型清理、PROJECTS/EXPLORATIONS/图谱更新。rqs若最终优于同容量linear，即便低于当前冠军，保留为局部机制；不能称当前组合提高。若无配对收益，停止当前PERM分位数/尾质量扫描；结论不泛化其他目标。训练中的暂时领先不算最终收益。
先启动linear观察实际显存后启动rqs，最多2任务不自动队列，记录单项与两任务实际总墙钟及并发/显存，含加载/预热/训练/评估，代码/最终重建另计。负权重删除先写清单，源配置紧凑证据保留。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成；无旧任务/其他折/交付覆盖。目标完成后独立井确认、新目录交付。


## v67 linear显式启动
{"pid": 30688, "mode": "linear", "started_at": "2026-09-29T22:40:21.759592+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_quantile_v67.py", "--mode", "linear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_quantile_v67_PERM_linear_fold5"], "automatic_queue": false}


## v67 rqs显式启动
{"pid": 7744, "mode": "rqs", "started_at": "2026-09-29T22:40:57.352614+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_quantile_v67.py", "--mode", "rqs", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_quantile_v67_PERM_rqs_fold5"], "automatic_queue": false}


## v67最终结果
两项完成，TOTAL=80.354325927816；rqs-linear PERM=+0.221226055194，训练后差=+0.221226055194；采用=None，保留=['rqs']。负权重清理，源/交付不改，无其他折；目标未完成，见perm_quantile_v67_20260929/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

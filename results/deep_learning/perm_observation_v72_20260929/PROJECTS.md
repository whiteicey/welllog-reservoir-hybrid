# v72 PERM有限观测分辨率

## 探索重点与对照
延续v67 RQS相对linear的局部收益+0.221226，其PERM88.827250仍低于当前89.396724。检验点似然是否追逐窄密度尖峰。本批point/interval均从v67第24轮完整838211参数和函数开始，全部参数训练；只改平均密度观测半宽0或0.05训练归一化单位。两组同五点Gauss-Legendre与logsumexp，默认BCE不变。是固定建模假设与近似积分，不宣称真实测量精度或官方效用。

## 已做与验证
两项已完成，均第18轮早停、最佳第0轮。训练后最高PERM分别88.778070945782、88.771566640436，区间组配对差-0.006504305345；含起点配对差0，没有采用。当前TOTAL80.354325927816保持。原9曲线206、原训练64井/第五16井144665点、Q50、1024中心128、原RQS解码保持；仅PERM训练标签，新AdamW3e-5/wd.01/clip2/cos80*64至1e-5/max80/pat18/min10/b64/64steps。初始状态和全点预测同一、四组真实梯度/最佳恢复/源码哈希/三列评分通过，详情见FINAL.md。

## 收益与不继续
没有本轮收益；停止当前观测宽度/积分节点扫描，不泛化所有标签噪声建模。已删除本轮4个无收益/重复权重共37411436字节，保留v67正控制、原模型与交付。实际串行峰值1，单项108.972/117.610秒、总墙钟260.439秒含启动间隔；加载预热训练评价计入，代码与收尾另计。

## 未完成与目标后
无其他折或旧队列。第五严格>80.5与完整原五折OOF>82仍未达成；接下来换机制，目标后独立井验证和新目录交付。PROJECTS/README/EXPLORATIONS/图谱已维护。
本文件修复原启动说明经PowerShell管道传输的中文编码损坏；不改已训练的源码、配置、分数和时间。


## v72 point显式启动
{"pid": 17092, "mode": "point", "started_at": "2026-09-29T23:32:33.373158+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_observation_v72.py", "--mode", "point", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_observation_v72_PERM_point_fold5"], "automatic_queue": false}


## v72 interval显式启动
{"pid": 14368, "mode": "interval", "started_at": "2026-09-29T23:34:51.127769+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_observation_v72.py", "--mode", "interval", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_observation_v72_PERM_interval_fold5"], "automatic_queue": false}


## v72最终结果
两项完成，TOTAL=80.354325927816；interval-point PERM=+0.000000000000，训练后差=-0.006504305345；采用=None，保留=[]。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见perm_observation_v72_20260929/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

# v35 进度

## 目标
在当前80.336513配方上检验跨井参考数值是否能改善SW连续预测，保留当前强模型。

## 已完成
重新连接D:/match_predict，读取最新约束，确认v34全部结束、没有活动训练；核对原评分器和冻结父模型接口。本批方案固定为一个SW任务中的matched/shuffled两个独立候选。

## 已固定方案
原64训练井每井确定性选择最多64个非默认SW参考点，使用训练侧既有幅值统计中的当前值/mean51/std51共27维做全库输入距离Top16。每个原始查询点排除自身整口井。神经模块根据当前SW隐藏信息、参考输入学习候选内注意力及混合比例；连续分布为原分布与参考标签核密度的凸组合，默认概率固定。参考标签仅来自训练井，负对照仅在同一参考井内打乱标签对应，防止隐含混入查询井标签。

## 未完成与计划
准备参考库与索引；运行两个同容量、同初始化、相同80/18预算的新候选；恢复与实际评分；对正候选才独立磁盘重建；写最终报告、来源及图谱。尚无新训练或新分数。

## 结论门槛与后续
matched必须超过原模型及shuffled才视为匹配信息的初步正证据；不因打乱对照偶然更好而部署它。该固定候选池仅学习重排序，未做全库动态度量学习。若无收益，不扫描K、参考库规模、核带宽或学习率。结果属于反复开发第五折，原其他四折和交付继续保持不动。完成后更新本文件及根PROJECTS，保留实际失败与有界负结论。


## v35显式启动
{"pid": 62064, "started_at": "2026-09-29T14:07:10.148637+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_retrieval_v35.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_retrieval_density_v35_SW_fold5"], "automatic_queue": false}


## v35门控失活诊断
2026-09-29T14:17:26.948609+08:00。已保存第18轮状态在64固定训练锚点7973有效点上，matched/shuffled门控proposal均100%负值，alpha与梯度全部为0；初始output梯度非零，但query/key零。18轮验证分数不变。不能将此结果当作充分训练后的检索负证据。旧源码/权重不改，新v36仅修复平滑门控通路。


## v35最终解释

原两项均第18轮完成，全部验证分数保持73.6630827970656。后续末态检查确认硬截断门控全关闭且梯度为0；不能将其作为充分训练后的跨井检索负证据。详见gate_diagnostic.json。历史源和checkpoint未改。v36用平滑门控作独立修正，详见../retrieval_smooth_gate_v36_20260929/FINAL.md。


Completed non-adopted checkpoints and neighbor index caches deleted after final recovery checks. Compact evidence retained; deletion manifests in v36 and v37 studies. Indexes require prepare_retrieval_v35.py to rebuild, not an available-cache claim.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

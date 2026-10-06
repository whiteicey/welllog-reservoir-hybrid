# 进度
目标：独立单任务模型，按通道选择有效结构。
已做：单/双私有骨干实现，六项有限对照已固定。
计划：分批训练、实际骨干梯度与最佳恢复、固定三模型组装验证。
未做：尚无本批精度收益或独立泛化结果，续跑等价性未实测。
完成目标后：比较模型成本并在固定配方下验证原分组泛化；当前不补折。


## 2026-09-28T11:11:39.865093+08:00 独立通道模型v8
六项预定义单/双骨干候选；本次实际启动[('POR_dual', 91632), ('SW_dual', 95292)]，各自端到端单通道训练。尚无本批已验证精度；保持原评分/分折，不自动补折或改变旧交付。计划和证据：results/deep_learning/independent_channels_v8_20260928。


## 启动失败及修复
首批POR_dual/SW_dual在main开头因遗漏sys导入失败，未加载数据或训练；已保存startup_failure_attempt1.json和原err。修复导入并明确重试，非自动恢复。另一次只读shell命令转义失败，不影响训练。结果以第二次实际日志为准。

## 独立模型实际结果
观察时间：2026-09-28T12:02:48+08:00。已完成6/6项。
已做：模型拆分和显式启动，结果如本目录progress.json。
计划：完成有限六项并固定三模型共同重载。
未做：完整OOF和独立泛化、续跑等价性。目标达成后验证固定配方，当前无补折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

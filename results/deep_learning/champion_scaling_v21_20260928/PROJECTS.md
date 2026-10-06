# v21进度

目标：解释80.274135的收益，完成冠军热启动下的内部宽度扩容对照。

已做：核对v1-v7实测报告、冠军结构和本地约束；确认本轮前无运行训练。POR贡献约98%的80.22至80.27增量；PERM贡献很小，SW不变。

计划：保存准确归因与原数据缓存；实现函数保持的1x/2x/4x候选；显式分批训练，完成同第五折预测组合和开销比较。

未做：扩容训练、后训练收益、模型加载验证；完整五折82目标未完成。

完成后：仅保留有实测收益的新开发配方；无收益保留v5/v8。不自动补折或覆盖交付。


## 2026-09-28T21:10:19.989243+08:00 v21显式启动
[{"mode": "base1", "pid": 38952, "started_at": "2026-09-28T21:10:19.980197+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "base1", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_base1_fold5"]}, {"mode": "ffn2", "pid": 42584, "started_at": "2026-09-28T21:10:19.987728+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_ffn2_fold5"]}]
仅预定义3规模之一或两项，最多2GPU，其他尚未启动项不自动排队。


## 2026-09-28T21:14:49.566168+08:00 v21显式启动
[{"mode": "ffn4", "pid": 50684, "started_at": "2026-09-28T21:14:49.561607+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "ffn4", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_ffn4_fold5"]}]
仅预定义3规模之一或两项，最多2GPU，其他尚未启动项不自动排队。


## v21冠军收益与扩容实验完成
2026-09-28T21:18:10.261733+08:00。1/2/4内部宽度三项训练/最佳恢复/实际预测组合均完成；采纳新POR=False，保留TOTAL=80.274135338。已完成收益来源/参数和开销记录；整网统一拓宽、其他目标扩容、其他折均未执行。证据champion_scaling_v21_20260928/FINAL.md。

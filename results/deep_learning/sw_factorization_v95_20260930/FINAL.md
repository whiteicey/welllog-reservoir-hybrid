# v95 固定概率分解结果

原第五TOTAL 80.442433883814 → 80.472576222471；采用=separate。

|固定密度来源|SW|TOTAL|对r100 SW|
|---|---:|---:|---:|
|same|73.965712645631|80.442433883814|+0.000000000000|
|separate|74.051833613221|80.472576222471|+0.086120967590|

探索重点：r100改善默认而数值退步，预先固定r100默认atom+自身/旧v82条件密度的两项同1678962容量消融；所有权重冻结，未训练。
实际验证：两个fold-local来源/源码/训练侧预处理/Q/原评分hash，原144665点原始输入评价，同来源重复分支与r100逐点相同。原概率公式和动作argmax，模型前向无验证标签、按点挑模型或预测投票。PERM/POR原两列固定。开发折机制消融不等于独立泛化或完整OOF。
收益：separate相对same SW=0.08612096759006249；默认/数值/井诊断={"scores": {"POR": 77.551936801885, "PERM": 89.39672404936582, "SW": 74.05183361322108, "TOTAL": 80.4725762224709}, "channel_delta": 0.08612096759006249, "TOTAL_delta": 0.03014233865651761, "populations": {"default": {"n": 100786, "reference": 98.45057848806643, "candidate": 97.99250476187036, "channel_delta": -0.4580737261960621, "TOTAL_contribution": -0.11169665433199953}, "numeric": {"n": 43879, "reference": 17.726242995100986, "candidate": 19.06232878431191, "channel_delta": 1.336085789210923, "TOTAL_contribution": 0.14183899298845698}}, "well_wins": 7, "well_losses": 9, "per_well": [{"well": 7, "n": 10219, "channel_delta": -0.19734447294429458}, {"well": 8, "n": 9011, "channel_delta": 0.6893414511690139}, {"well": 14, "n": 12078, "channel_delta": -0.4788646883792609}, {"well": 16, "n": 8149, "channel_delta": -0.2931756625087303}, {"well": 25, "n": 7739, "channel_delta": 0.461190122361387}, {"well": 34, "n": 10913, "channel_delta": -0.0518393898189089}, {"well": 35, "n": 8348, "channel_delta": 2.0295073731362265}, {"well": 37, "n": 9939, "channel_delta": -0.6393305447997051}, {"well": 50, "n": 11874, "channel_delta": 0.819757462626086}, {"well": 54, "n": 9630, "channel_delta": 0.708406095439372}, {"well": 63, "n": 5639, "channel_delta": -0.9521693103723123}, {"well": 69, "n": 9120, "channel_delta": -1.0792803736472507}, {"well": 71, "n": 4709, "channel_delta": -0.08720162817995458}, {"well": 72, "n": 9559, "channel_delta": 0.10985075663464272}, {"well": 76, "n": 7109, "channel_delta": 0.7734814079279602}, {"well": 79, "n": 10629, "channel_delta": -0.4362100253159724}], "exploratory_selection_unadjusted_TOTAL_95": [-0.0958218812329302, 0.16261901873112136]}
不继续：若负，关闭这两个固定来源直接拼接的结构；不扫多个来源或比例。旧v84r1是旧gate/私有密度热启动训练best0负结果，本批新SAM gate+已有固定density零更新对照，结论范围不同。
已做：固定两项结构、原始输入评价、实际三列、来源保留。未做：新训练、其他折、独立井泛化。第五严格>80.5且有收益后才考虑补折，原完整OOF>82未完成，不自动启动。目标后独立确认与新目录交付。
成本：{"candidates": [{"mode": "same", "seconds": 3.696581900003366, "peak_allocated_MiB": 436.56396484375, "peak_reserved_MiB": 502.0}, {"mode": "separate", "seconds": 2.6655303000006825, "peak_allocated_MiB": 436.56396484375, "peak_reserved_MiB": 502.0}], "total_wall_seconds": 7.497336500004167, "actual_gpu_process_concurrency": 1, "scope": "train-only cache/input bank loading, both frozen two-network original raw-input evaluations and persistence included; no training/warmup exclusion; authoring/atlas separate"}
未新增/复制checkpoint；两来源仍是当前/正控制依赖，保留。没有负权重大文件需要删除。

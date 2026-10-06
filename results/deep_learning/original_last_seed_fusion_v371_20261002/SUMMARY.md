# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# 原模型与最后一种子联合预测

目标：仅对原模型与已完成 seed20261014 做第五折概率分布融合。无新训练。保留补折原模型+seed13计划。
约束：原分折/144665行/原解码/normal GPU/最多2任务；单个头所有行共用一个权重，禁止逐行和按标签类型选模型。每批维护PROJECTS.md。

探索重点：0:0.01:1权重扫描，比较等权、最佳共同权重和三个头各自最佳权重。
已探索与验证：入口已准备，当前尚无融合成绩；需原生磁盘恢复预测/端点/全行覆盖与CPU三头组装。
收益：尚未证实，不混同完整五折OOF。用同一第五折选择权重的得分是调参诊断。
不值得继续：无证据；不扩大为其他配对搜索或改变补折模型。
未完成：三个目标原生读出、融合曲线、结果表和图谱登记。
计时：分别记录各头加载/预热/推理/扫权重/写盘，以及完整三头总墙钟；串行，不把max2当默认双并发。

V371 final per-head diagnostic TOTAL80.849582637629; delta-original+0.143452914391; delta-original+seed13-0.013874919753. Native disk replay, endpoints, coverage and actual three-head assembly/readback exact. No extra training.




## RESULTS.md

|Method|POR|PERM|SW|TOTAL|
|---|---:|---:|---:|---:|
|Original|78.046431|89.518389|74.173612|80.706130|
|Seed14|77.420328|89.449227|73.796077|80.361955|
|Equal 50:50|78.202819|89.699768|74.214195|80.830733|
|Common best|78.189359|89.713476|74.243973|80.841915|
|Per-head best|78.205648|89.714844|74.250551|80.849583|

Weights (original/seed14): {'POR': {'weight_original': 0.52, 'weight_seed14': 0.48, 'score': 78.20564817956807, 'index': 52}, 'PERM': {'weight_original': 0.59, 'weight_seed14': 0.41000000000000003, 'score': 89.71484414555762, 'index': 59}, 'SW': {'weight_original': 0.6, 'weight_seed14': 0.4, 'score': 74.25055066518057, 'index': 60}}

Same fifth tuning diagnostic. FullCV remains original+seed13.





# 第五折冲 80.4 + test 伪标签验证计划

> 分支:whiteicey/welllog-reservoir-hybrid @ deep_learning
> 本地克隆:E:\Temp\welllog-repo(已切到 deep_learning,FETCH_HEAD=f07a7d7)
> 数据:D:/match_predict/data(train/*.txt),训练入口 tools/deep_learning/train.py
> 编写时间:2026-09-26,仅规划,未执行任何训练

## 0. 现状(事实基线)

| 项 | 数值 | 出处 |
|---|---|---|
| 已交付完整 OOF | 81.7132(五折,A 榜用户报 83.6915) | results/deep_learning/full_oof_original_verification.json |
| 当前第五折最好 | 80.2509(posterior-router,未过 80.4) | results/deep_learning/final_posterior_router.json |
| 新晋级门槛 | 严格 >80.4 | tools/deep_learning/README.md |
| 缺口 | 80.4 − 80.2509 ≈ **0.149** | — |
| 训练预算 | 80 epochs / patience 18 / seed 20261006 | tools README |
| DL 线伪标签 | **从未做过**(仅树线 v3 用过) | grep 证实 |

## 1. 用户三个想法的解读

1. **只打第五折、过阈值再补其余折** —— 已是现状,fold-count 1 硬约束,没问题。
2. **第五折先超 80.4** —— 主线任务,下面 Phase A。
3. **把 test 标个 validate 出来:打多版伪标签→提交→按评测分挑最准版→加进训练** —— 即「LB 反馈驱动的伪标签筛选」,下面 Phase B/C,重点控风险。

## 2. Phase A:第五折冲 80.4(主线)

缺口只有 0.149,后验路由(80.2509)和局部路由(80.2219)是两个**独立**改动,尚未叠加。优先做组合,而不是开新方向。

### A1. 组合实验:局部卷积残差 + 后验描述量路由(最高优先)

- 依据:local-router +0.0507(对逐点路由),posterior-router +0.0290(对局部路由),两者正交,叠加期望 80.28~80.33,接近但可能仍差一口气,需再加一档改进。
- 文件:新建 `experiments/pure_dl/train_paired_local_posterior_router80_sequence.py`(由 posterior-router 版 + local-router 的零初始化局部卷积残差合并而成)。
- 冻结:80 epochs / patience 18 / lr 0.0006 cosine / seed 20261006 / 固定第五折(seed+4)。
- 运行:`conda activate normal && python tools/deep_learning/train.py --method <新注册名> --data_dir D:/match_predict/data --out outputs/local_posterior_router_01 --fold-count 1`(需先在 tools/deep_learning/train.py 注册新 method 映射)。
- 判定:用未四舍五入的原评分,严格 >80.4 才晋级;80.30~80.40 之间加跑一个种子再议;<80.30 记为负结果。

### A2. 若 A1 未过线的备选(按序尝试,每次只动一个变量)

1. **三种子多数投票路由权重**:posterior-router 结构不变,跑 seed+0/+1/+2 三个种子,第五折推理时三路路由后验等权混合。成本约 3×18 分钟(GPU),风险低。
2. **SW 通道针对性**:后验路由 SW=73.67 是三条通道里最弱的;检查 SW 默认值边界(≥99.9)分类头在第五折的错分,考虑 SW logit 坐标回头试(历史负结果 79.65 但那是旧基线上的,新基线未必)。
3. **窗口长度**:256 点窗口换 384/512,只改一处,看第五折反应(注意历史负结果多在首折,不直接迁移结论)。
4. **特征侧增量**:用户提到"通过做特征已经取得了很好的收益"(树线)——把树线 v3 里最有效的井内统计特征(如分位数分箱边际)挑 2~3 个喂给 68 维视图成员,每次加一个。

### A3. 过线之后

- 严格 >80.4 后,按原流程申请补折授权,完成五折完整 OOF 核验,再考虑提交。
- 全程不篡改历史筛选分,不拿单折分数宣称泛化收益。

## 3. Phase B:test「盲标撞答案」可行性设计(谨慎)

### B1. 风险先讲清

- **赛规风险**:人工构造 test 标签提交,多数比赛属违规或灰色;先核对赛规"禁止事项"再动。
- **过拟合 public LB**:5 口井、提交次数有限,分数反馈是 1 比特/次的黑盒信号,刷多了 private 必崩。
- **信息论现实**:一次提交只能拿回一个标量,想靠它从多版伪标签里挑"最准"的,版数不能多(≤4 版),且每版之间要有实质差异。

### B2. 如果要玩,这么玩(预算 3~4 次提交封顶)

1. **生成 3 版伪标签**,差异来自"可信度分档"而不是随机扰动:
   - V1 保守:只标当前五折集成对 test 预测置信度最高的 20% 点(后验混合权重熵低 + 三通道默认值概率离 0.5 远),其余点标 NaN(不参与训练)。
   - V2 中等:置信度前 50%。
   - V3 激进:全部 5 口井 95,948 行全标。
2. **不分别提交 3 版**(浪费次数)。先本地用"折间一致性"做免费筛选:拿其中一口 test 井的伪标签当假 valid,看模型在其余 4 口上的表现是否单调提升,挑本地信号最好的一版。
3. **只提交本地信号最好的 1 版**,记录 LB 分数与本地信号的对应关系。
4. **停止线**:若提交分数 ≤ 当前 A 榜 83.6915,立即停手,说明伪标签方向在 DL 线上无收益或已被 LB 噪声淹没;不追加次数。

### B3. 与树线 v3 的教训对照

v3 的伪标签经验(pseudo_weight 0.3~0.5、SW 翻转率 >0.03 就不稳)直接复用:DL 版伪标签同样按通道分开处理,SW 单独设更严的纳入门槛,POR/PERM 可以放宽。

## 4. Phase C:test 作 validate(伪标签的正确姿势)

这是 Phase B 的收敛版,也是更值钱的长期资产:

1. **双循环**:外循环用伪标签版训练→对 test 重新预测→只更新置信度排序变化的点,迭代 2~3 轮即停(防漂移)。
2. **validate 用法**:test 伪标签不进训练 loss,只当早停/选 epoch 的验证信号(替代现在"本折验证选 best epoch"的选择偏差),这样不直接污染训练。
3. **证据记录**:每版伪标签的生成规则、置信度阈值、LB 分数写进 results/deep_learning/ 下的新 json,与 OOF 严格分开标注"test 标签不可得,分数为用户报分"。

## 5. 文件改动预估

- 新增:`experiments/pure_dl/train_paired_local_posterior_router80_sequence.py`(A1)
- 修改:`tools/deep_learning/train.py` 注册新 method
- 新增:`experiments/pure_dl/pseudo_label_*.py`(B/C,需新授权后动工)
- 新增:`results/deep_learning/pseudo_label_log.json`(B/C 证据)
- 不动:任何已冻结训练文件的模型/损失/特征/种子(遵守 tools README 的冻结承诺)

## 6. 验证与门槛

| 阶段 | 通过条件 |
|---|---|
| A1 单折 | 第五折 TOTAL 严格 >80.4(未四舍五入原评分) |
| A 补折 | 完整五折 OOF >81.7132 且三通道无翻转率异常 |
| B 伪标签 | LB 分数 >83.6915 才继续,否则一轮停 |
| C validate | 伪标签选出的 epoch 与真实 best epoch 的 Spearman 相关 >0.8 |

## 7. 风险与未决问题

- **GPU 排队**:guard 限 2 作业,训练 18 分钟/次,A1+A2 候选排队约 1~2 小时。
- **赛规未核对**:Phase B 动工前必须先确认"test 伪标签提交"不违规,否则整 Phase B 砍掉。
- **0.149 缺口的置信度**:单种子噪声量级未知,若 A1 落在 80.35~80.40,需 2 种子确认,不接受擦线晋级。
- **新授权**:Phase B/C 涉及新训练,按仓库惯例需要用户明确授权后才启动。

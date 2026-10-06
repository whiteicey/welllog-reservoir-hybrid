# v48 条件分位密度：线性与平滑样条

## 探索重点与假设
v47任务记忆无收益，当前保留v46 TOTAL80.342771761689、SW73.680963725271。本批更换连续输出分布机制：v46共享骨干/整井FiLM/默认状态头热启动，原8高斯混合密度删除，换为神经网络预测33个有序分位点；全骨干与新头均训练。原默认概率仍与连续密度共同解码，不混合旧预测或按标签选点。
两项预定义linear/rqs，同838211参数和同初始状态。新密度头34输出：中点、总范围、32个间距logits，softplus范围与softmax间距保证有序，间距下界.001。固定CDF高度从.005至.995等距；两端用斜率匹配的指数尾延伸至全实数，不硬裁标签。linear在相邻分位点间用线性CDF；rqs用单调有理二次CDF，内节点导数由相邻割线斜率调和平均确定，不加额外参数。
公式来源：[Durkan等，Neural Spline Flows (2019)，附录A公式19/22](https://arxiv.org/html/1906.04032)。本项目固定CDF高度、调和导数与指数尾是明确的适配，不是论文coupling-flow基准复现。按CDF对标签的导数计算完整密度负对数；在原变换标签空间建模，原标签变换不变。
假设更灵活的条件密度能刻画SW连续值形状，而非继续增加输入记忆。对照可检验同分位参数下插值平滑性的收益；不假定样条优于混合分布。先报告新分布初始成绩，再报告训练后最佳与旧SW差，不能把初始化差当训练收益。

## 初始化与数据边界
骨干/FiLM/默认头加载v46自身原训练折权重，属于热启动，非外部预训练。新密度零权重、偏置来自原64训练井非默认标签的经验分位数（不加权初始分位；训练损失仍用原Q50）。新分布不保持原函数；最佳候选从自身第0轮及实际训练轮中选择，是否替换旧模型单独判定。当前井输入库无标签/井ID，16验证井库只用于本井推理；标准化、标签变换、网格和Q均复用原训练侧来源。
固定原第五144665点，seed20261010，原9曲线206输入，1024窗口/中心128，原256支持点与800动作加默认值。全CDF质量在原支持边界相减并归一，与原网格规则一致。训练仍为Q加权默认BCE加非默认密度NLL，除3保持原尺度；无效点或默认行不贡献连续损失。

## 训练、已做和待做
已实现两种CDF/密度、完整骨干训练、原评分解码和显式启动器；启动前无成绩。两项均max80/pat18/min10/b64/64步，AdamWlr1e-4/wd.01/clip2、cosine80*64至1e-5，原子last_state保存优化器/调度/RNG/停机状态，显式恢复。最多2GPU，无自动队列。
直接实际训练，记录初始/训练快照/最终恢复及真实梯度；新增分布的CDF单调性和概率合法性随真实评分检查。结尾实际三列评分；新正候选才增加全144665点磁盘重建，原已验证预测不重复跑。CUDA注意力/插值反向warn_only，不声称重新训练逐位确定。
待做两项训练、同容量结果比较、实际组合、紧凑诊断、负权重清理和图谱更新。若均不提高，不继续扫分位点数或尾部质量。单项耗时与双并发总墙钟/显存分开，数据加载/预热/输入库计入；代码编写与最终重载另计。
原第五严格>80.5并有明确收益后才考虑其他折；完整OOF>82未完成。无其他折/旧队列/交付覆盖；维护PROJECTS目标/已做/未做/后续，独立README、EXPLORATIONS及atlas。目标达成后的独立井确认和新目录交付尚未开始。


## v48 linear显式启动
{"pid": 57888, "mode": "linear", "started_at": "2026-09-29T17:31:15.412049+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_density_v48.py", "--mode", "linear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_density_v48_SW_linear_fold5"], "automatic_queue": false}


## v48 rqs显式启动
{"pid": 60688, "mode": "rqs", "started_at": "2026-09-29T17:31:40.553599+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_density_v48.py", "--mode", "rqs", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_density_v48_SW_rqs_fold5"], "automatic_queue": false}


## v48任务查询对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后rqs-linear=+1.292082008243。详见quantile_density_v48_20260929/FINAL.md；无其他折或交付修改。

## v48最终收尾：保留有对照收益的平滑密度
两组新密度初始SW均72.346441547110，非父模型73.680963725271。linear训练后最高第7轮72.136427521820，第18轮早停、回退自身第0轮；rqs训练后最高第4轮73.428509530063，第22轮早停。RQS较linear含初始最佳+1.082067982952SW（等价TOTAL+0.378723794033），仅比较训练后最高差+1.292082008243；这两个差值不能混用。
同容量/同初始化对照13井升3井降，选择未校正的探索性井级TOTAL区间[0.163806,0.602874]。因此保留RQS模型作为有局部机制收益的候选；不因绝对低于当前SOTA而删除。RQS实际替换组合TOTAL80.254412793365，仍低当前80.342771761689（SW差-0.252454195209），不采用进当前配方、不补折。完整OOF>82仍未完成。
新分布解析密度在8个实际训练锚点输出上与CDF自动微分相符，最大相对误差linear5.94e-7/rqs8.38e-7，全实数CDF端点精确0/1；这是新数学实现的单次最终检查，不是重复预检。RQS最佳权重磁盘重建全144665点与存档完全一致，证据retained_rqs_verification.json。原最终验证已恢复各自最佳；原三目标保留组合实际评分完成。
单项158.51/183.54秒，双并发总墙钟213.127631秒，含加载/预热/输入库/启动间隔；峰值各1261.56MiB、max2GPU。代码编写与最终RQS重载检查2.21秒另计。两项已结束，无自动队列。删除linear两个负权重，保留RQS最佳/完整状态、v46正模型、父模型和旧交付；哈希/字节见deletion_manifest.json。
不继续当前线性密度、分位点数或尾概率扫描。后续可带着已证实的平滑插值机制，独立研究分位位置的监督或分布训练方法；这只是下一步方向，尚未训练，不宣称已取得新收益。目标、剩余工作及完成后独立井确认/新目录交付继续维护。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

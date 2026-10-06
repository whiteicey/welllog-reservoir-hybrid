# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v66 SW整井层序残差

## 探索重点与预定义对照
当前v56 SW73.713975628494、组合80.354325927816。旧整井编码对128个深度均匀采样点仅取64维隐表示的均值/标准差/最大值，忽略顺序。本轮冻结整个原预测器，仅加85216参数整井条件残差。先经原冻结point编码得到128×64，再Conv1d64→32/kernel3/GELU，汇总32均值和32标准差，Linear64→1216给六个FiLM层输出0.1*tanh增量。旧FiLM保留，新增输出权重/偏置全零，初始预测不变；原±0.5界加新增±0.1，整体界不冒称仍为±0.5。
预定义permuted/ordered两项，同参数、全部张量与初始化；permuted固定同一随机排列128点、ordered原深度排列。排列只由seed20261166产生，所有训练和验证井共享，不读标签/井ID；只改变邻近点组合，信息集合、pooling/核宽度相同。不混入其他模型预测。

## 数据训练边界与已做
原9曲线/206输入、64训练井预处理/Q50、第五16井144665点、seed20261010、1024随机窗口中心128、原Gaussian+BCE SW损失/解码/官方评分保持。全井输入库无标签，验证井仅自己的推理输入；原骨干/输出头/FiLM全冻结但新调制到输出的梯度通路保留。仅新支路AdamW3e-4/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps。相同dropout eval，原函数纳入选择，回退不是学习收益。
已实现，启动时无成绩。直接训练，实际检查新参数梯度、旧权重不变、源哈希、初始/最佳144665点及最终三列分数，不做独立预检/审查。保留新模型前原始输入磁盘重建。完整原子last_state保存网络/优化器/调度/RNG/早停，仅显式恢复。

## 待做、收益与不继续
两项实际训练、同容量顺序收益和绝对差、负权重清理、图谱及PROJECTS待完成。先启动一项观察真实显存余量，再决定第二项并发或串行；max2不是默认2，不自动队列。分别记录单项和同等任务总墙钟，含加载/预热/训练/评估，代码/最终重建另计。
若ordered没有最终配对收益，不扫本支路核宽度/残差幅度；只限当前稀疏采样层序编码，不泛化所有地层顺序信息。第五严格>80.5且明确收益后才考虑补折，完整原五折OOF>82和独立确认未完成，无旧队列/其他折/交付覆盖。维护PROJECTS/EXPLORATIONS/图谱；目标完成后独立确认、新目录交付。


## v66最终结果
两项完成，TOTAL=80.354325927816；ordered-permuted SW=+0.000000000000，训练后差=+0.007728273109；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_well_order_v66_20260929/FINAL.md。




## FINAL.md

# v66 SW整井层序残差结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|permuted|73.576599127307|73.713975628494|0/18|
|ordered|73.584327400417|73.713975628494|0/18|

## 探索重点与验证
同924697总参数、85216可训练参数和完整初始函数；原v56全预测器冻结。原point隐表示经kernel3 Conv64→32、均值/标准差池化、零初始Linear64→1216生成六层FiLM增量，0.1*tanh边界。permuted读固定随机次序，ordered读深度次序；输入点集合、参数、初始化相同。原主损失/Q50/1024上下文/解码不变。旧0.5与新0.1边界分别记录。
完整初始状态/核心/输出头/144665点预测相同；实际新增Conv/输出梯度有效，全部原参数无梯度且训练结束哈希不变。最佳恢复、源哈希和三列评分已核验。保留者原始输入磁盘重建=False。

## 收益与不继续
ordered-permuted含回退SW差+0.000000000000；训练后最佳差+0.007728273109；采用=None。
回退第0轮不是学习收益。若ordered无最终配对收益，不继续本整井卷积核宽度/残差幅度扫描；结论只限当前128点稀疏层序支路，不泛化全部层序信息。重复开发折未经独立新井确认。

## 耗时、清理与未完成
{
  "candidates": [
    {
      "mode": "permuted",
      "start": "2026-09-29T22:33:06.556430+08:00",
      "end": "2026-09-29T22:34:35.717051+08:00",
      "seconds": 85.21408549998887,
      "peak_allocated_MiB": 1180.66162109375
    },
    {
      "mode": "ordered",
      "start": "2026-09-29T22:34:43.129008+08:00",
      "end": "2026-09-29T22:36:13.107225+08:00",
      "seconds": 85.63762580000912,
      "peak_allocated_MiB": 1180.66162109375
    }
  ],
  "first_launch_to_last_completion_seconds": 186.550795,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.34030699999129865,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final r

# v66 SW整井层序残差结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|permuted|73.576599127307|73.713975628494|0/18|
|ordered|73.584327400417|73.713975628494|0/18|

## 探索重点与验证
同924697总参数、85216可训练参数和完整初始函数；原v56全预测器冻结。原point隐表示经kernel3 Conv64→32、均值/标准差池化、零初始Linear64→1216生成六层FiLM增量，0.1*tanh边界。permuted读固定随机次序，ordered读深度次序；输入点集合、参数、初始化相同。原主损失/Q50/1024上下文/解码不变。旧0.5与新0.1边界分别记录。
完整初始状态/核心/输出头/144665点预测相同；实际新增Conv/输出梯度有效，全部原参数无梯度且训练结束哈希不变。最佳恢复、源哈希和三列评分已核验。保留者原始输入磁盘重建=False。

## 收益与不继续
ordered-permuted含回退SW差+0.000000000000；训练后最佳差+0.007728273109；采用=None。
回退第0轮不是学习收益。若ordered无最终配对收益，不继续本整井卷积核宽度/残差幅度扫描；结论只限当前128点稀疏层序支路，不泛化全部层序信息。重复开发折未经独立新井确认。

## 耗时、清理与未完成
{
  "candidates": [
    {
      "mode": "permuted",
      "start": "2026-09-29T22:33:06.556430+08:00",
      "end": "2026-09-29T22:34:35.717051+08:00",
      "seconds": 85.21408549998887,
      "peak_allocated_MiB": 1180.66162109375
    },
    {
      "mode": "ordered",
      "start": "2026-09-29T22:34:43.129008+08:00",
      "end": "2026-09-29T22:36:13.107225+08:00",
      "seconds": 85.63762580000912,
      "peak_allocated_MiB": 1180.66162109375
    }
  ],
  "first_launch_to_last_completion_seconds": 186.550795,
  "peak_process_concurrency": 1,
  "finalization_seconds": 0.34030699999129865,
  "scope": "loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate"
}
本轮删除4个负检查点，共27892276字节，清单先写后删除；源、配置、日志、预测与曲线保留。
并发峰值与真实墙钟见timing；单项含加载/预热/训练/评估，总墙钟含显式启动间隔，代码和收尾另计。CUDA反向允许非确定，不声称逐位重训复现。第五严格>80.5及完整原五折OOF>82仍未完成；无其他折、旧队列或交付修改。达成后独立确认及新目录交付。



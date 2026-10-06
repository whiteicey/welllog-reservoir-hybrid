# v91 冻结当前Gaussian的全井直接头

探索重点：v90同容量well-reference SW+0.004915085801，完整两组都低于Gaussian，原模型保持。把这个局部证据放到当前v82 a100完整函数上，冻结全部839481原参数，仅新增2425参数零Linear96→25，避免全骨干重学和先验重置损失。两组仅新头读取本井code或训练64井参考code不同，同初始化/容量/梯度图/计算；原FiLM两组均本井，reference只用训练测井输入均值。不会加载v90训练后负绝对权重。

对照：当前SW73.793310952488、固定POR/PERM TOTAL80.382093291214。预定义well-reference训练后最佳差与相对当前绝对差分别报告。初始完整144665动作须同当前；原9曲线206+深度/原64-16/Q50/seed20261010/256支持800动作及默认99.9不变，BCE+GaussianNLL+联合效用1/3，温度.05。冻结base eval，DropKey不激活；与v90全网络DropKey训练不同，不能直接以v90-v91差归因。两头各独立AdamW1e-4/wd.01/clip2/cos80*64到1e-5/max80pat18min10/batch64/64步。共享冻结提取，每组对自身SW选模/早停。

已探索与验证：建立时未训练，收益尚未证实。实际训练检查新头与效用梯度、冻结无梯度/最终权重无变化、源/评分/预处理、初始全预测和最佳恢复。正绝对或完整公平局部收益重载头与父模型原始输入；实际三通道组合，不能将epoch0恢复当新收益。无需独立预检/重复Q。

收益与不继续：完成后报告绝对/匹配、默认/数值/各井与实际停止；若无绝对和well-reference训练后收益，关闭当前冻结Gaussian/共享96code/零线性25头/1e-4/预算组合，不临时扫宽度/LR/强度。局部正可带入隔离新假设，但非SOTA或独立泛化。原子完整last，恢复显式；训练后最佳state/pred保留至对照评估。

已做：两有限头和共享训练入口。计划：实际训练/组合/必要重载/清理/图谱。未做：真实收益/其他折/独立泛化。一个GPU进程，实际加载预热/训练验证保存/单项共享工作量/总墙钟和显存分别记录，不把并发上限当默认。目标后：原第五严格>80.5且明确收益再考虑其余四折，完整OOF>82后新目录交付和独立井确认；旧交付保持。


## 最终结果
# v91 SW冻结全井直接头结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|SW|固定POR/PERM TOTAL|最佳/停止轮|训练后最高SW|
|---|---:|---:|---:|---:|
|reference|73.793310952488|80.382093291214|0/18|73.706042498625|
|well|73.793310952488|80.382093291214|0/18|73.614141362181|

## 探索与验证
同2425零Linear96→25头、同当前Gaussian初始全预测；原v82最佳25轮完整SW模型839481参数固定，原FiLM两组读本井。新支路只差训练64井参考code/当前本井code，两code都计算；共享冻结前向eval无DropKey。原9曲线206/输入bank/分折/Q50/原256支持800动作不改，仅SW BCE+GaussianNLL+效用1/3，温度.05，新head1e-4/cos至1e-5/80上限pat18。与v90全网络训练不同，不能将两批差归因单因素。
实际第二batch头/效用梯度非零，来源/预处理/source hash、冻结无梯度/终态不变、初始和最佳144665点恢复、原行序和三列评价通过；正候选磁盘重建=[]。

## 收益与不继续
well-reference含初始最终SW +0.000000000000；训练后最高差 -0.091901136443。采用=None，保留=[]/训练后对照=[]。epoch0恢复不算新收益，完整公平训练后最佳差分开报告，不替代当前模型选择；正增益两组更新后head/pred必须磁盘重载。
reference 默认/数值/井变化：{"default": {"n": 100786, "reference": 97.48140224897922, "candidate": 97.37207168732378, "channel_delta": -0.1093305616554403, "TOTAL_contribution": -0.0266591538758637}, "numeric": {"n": 43879, "reference": 19.383958656215817, "candidate": 19.347364821096935, "channel_delta": -0.03659383511888237, "TOTAL_contribution": -0.003884804976417957}}；7胜/9负。
well 默认/数值/井变化：{"default": {"n": 100786, "reference": 97.48140224897922, "candidate": 97.23475423704699, "channel_delta": -0.24664801193223695, "TOTAL_contribution": -0.060142628042103144}, "numeric": {"n": 43879, "reference": 19.383958656215817, "candidate": 19.35978075218136, "channel_delta": -0.024177904034456077, "TOTAL_contribution": -0.00256672856526986}}；6胜/10负。
若无绝对或预定匹配收益，关闭当前冻结Gaussian/共享96code/零25输出/1e-4/预算组合，不临时扫宽度LR强度，不泛化全部全井机制；单开发折选择非独立泛化确认。

## 成本、清理、未完成与目标后
{
  "shared_process_seconds": 133.46452280000085,
  "launch_to_last_completion_seconds": 137.87569,
  "launch_to_verified_seconds": 287.491235,
  "peak_training_processes": 1,
  "peak_allocated_MiB": 1119.74853515625,
  "peak_reserved_MiB": 1304.0,
  "epoch_timing": [
    {
      "epoch": 1,
      "seconds": 7.256077700003516,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 2,
      "seconds": 6.618718499987153,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 3,
      "seconds": 6.854741999995895,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 4,
      "seconds": 6.674110499996459,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 5,
      "seconds": 6.811326199996984,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 6,
      "seconds": 6.846612399996957,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 7,
      "seconds": 6.712326700013364,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 8,
      "seconds": 6.731535500002792,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 9,
      "seconds": 6.806745600013528,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 10,
      "seconds": 6.817162199993618,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 11,
      "seconds": 6.758336300001247,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 12,
      "seconds": 6.762468500004616,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 13,
      "seconds": 6.8469656000088435,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 14,
      "seconds": 6.583344999991823,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 15,
      "seconds": 6.708985400007805,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 16,
      "seconds": 6.538429499982158,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 17,
      "seconds": 6.655808300012723,
      "active_modes": [
        "reference",
        "well"
      ]
    },
    {
      "epoch": 18,
      "seconds": 6.56886949998443,
      "active_modes": [
        "reference",
        "well"
      ]
    }
  ],
  "finalize_seconds": 0.6884245999972336,
  "scope": "two independent2425heads sequentially update in one GPU process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate"
}
清理5个无亮点/冗余状态，共8118892字节；正源和交付不改。
未运行其他折或旧队列。原第五严格>80.5且明确收益后才考虑其余四折；完整原OOF>82未完成。目标后独立井验证和新目录交付。


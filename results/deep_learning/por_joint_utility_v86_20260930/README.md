# v86 POR似然与联合动作效用

## 探索重点与固定对照
SW输出MLP64/局部3点都未取得收益，停止该结构批次，保留当前独立路线TOTAL80.382093291214。转向POR77.551936801885的v31 anchored完整函数，同30819个既有路由参数、同初始化、同输入/损失图/随机窗口；原专家、骨干、位置支路、锚点冻结，只有路由优化器更新，无新容量。
预定义likelihood/joint两项：BCE+Student-t3 NLL（原Q50、除3）与同一似然加系数1的联合动作效用（Q50、除3），动作softmax温度0.05。两组均计算效用图，系数0/1为唯一组间差异。先验和起点不重置，来源v31 anchored第7轮；原优化器不加载。
POR使用标准差参数化的t3 CDF、训练归一化log1p的逆expm1、官方相对容差0.08、默认0.1。原256支持格/800几何动作+默认/原singlecoverage解码完全保持；可微CDF先替换无穷尾再恢复常数，512点梯度checkpoint减少中间显存。效用是argmax代理，不能把它称为精确官方优化。

## 已探索与实际验证
本记录建立时未启动，收益尚未证实。训练直接运行，无独立数值预检或审查。实际起点全144665预测和当前POR一致；实际第二batch记录似然总梯度和效用独立梯度，防止效用无梯度。最终检查冻结状态、最佳恢复、来源hash、原行序和真实固定三列评价；正候选再原始输入磁盘重建。
原64训练/16验证井、578891训练点、144665第五折、seed20261010、原9曲线206及既有幅值63、Q50保持，统计不重新看验证标签；训练仅POR标签，不使用其他目标真值或模型预测。PERM/SW固定现有选择。新AdamW3e-5/wd.01/clip2/max80/pat18/min10/b64/64步，cos80*64至1e-5，这是新优化轨迹；当前主模型和历史脚本不改。原子last包含两个头/各自优化器/调度器/RNG/停止状态，恢复显式授权。

## 收益与不值得继续
最终分别报告两项对当前POR绝对差、joint-likelihood匹配差、默认/数值/每井变化和三列TOTAL。若匹配损失有稳定实测增益但未超过当前绝对分，保留机制并如实区分，不把epoch0恢复算新收益。若两项均无训练收益/匹配效用收益，则关闭此源/路由范围/温度/系数，而不临时扫系数或延长轮数。当前尚无结论，不泛化所有效用学习。

## 已做、计划、未做与目标后
已做：独立协议、模型/损失/训练入口。计划：只运行这两个头，实际评分/正候选重建/负权重清单/图谱。未做：真实收益、其他折与独立泛化；当前不补折。一个GPU进程共享冻结提取、两头依次独立更新，并非两个GPU进程，最多2GPU限制保持；单臂不能从共享总耗时伪造，逐轮活跃臂/过程墙钟/预热数据和显存如实记录。目标后：达到第五严格>80.5且明确收益再决定原其他折，完成原OOF>82后新目录交付和独立井验证；旧模型/交付保留。


## 最终结果
# v86 POR联合动作效用最终结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|POR|固定PERM/SW TOTAL|最佳/停止轮|训练后最高POR|
|---|---:|---:|---:|---:|
|likelihood|77.551936801885|80.382093291214|0/18|77.549090528980|
|joint|77.551936801885|80.382093291214|0/18|77.548343710813|

## 探索与验证
同30819既有anchored路由、同起点，原专家/骨干/位置头冻结，无新增容量。效用只读POR训练标签，t3安全尾CDF/原动作/0.08容差/log1p逆变换保持；0/1系数、温度.05、Q50、80上限/pat18，新3e-5及cos到1e-5。
实际第二batch效用梯度非零，来源/预处理/source hash、冻结状态、最佳恢复、144665点原行序和三列评价通过；正候选磁盘重建=[]。

## 收益与不继续
joint-likelihood最终POR +0.000000000000；训练后最高差 -0.000746818167。采用=None，保留=[]。epoch0恢复不算新收益，更新后最高为独立诊断，不替代最终选择。
likelihood 默认/数值/井变化：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 97.22649088037983, "channel_delta": -0.0010598047839636138, "TOTAL_contribution": -0.0002073756610108624}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.6424262816036, "channel_delta": -0.006196948195544394, "TOTAL_contribution": -0.0006465062105307725}}；8胜/8负。
joint 默认/数值/井变化：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 97.22755068516379, "channel_delta": 0.0, "TOTAL_contribution": 0.0}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.638290986021524, "channel_delta": -0.010332243777618544, "TOTAL_contribution": -0.001077927321669582}}；6胜/9负。
若无绝对或预定匹配收益，关闭此来源/路由范围/系数1/温度.05组合，不泛化全部效用学习；单开发折选择非独立泛化确认。

## 成本、清理、未完成与目标后
{
  "shared_process_seconds": 408.190708500013,
  "launch_to_last_completion_seconds": 413.258537,
  "launch_to_verified_seconds": 606.777047,
  "peak_training_processes": 1,
  "peak_allocated_MiB": 752.11669921875,
  "epoch_timing": [
    {
      "epoch": 1,
      "seconds": 22.255112599988934,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 2,
      "seconds": 22.05823419999797,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 3,
      "seconds": 22.11560930000269,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 4,
      "seconds": 22.20747350002057,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 5,
      "seconds": 22.19094710002537,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 6,
      "seconds": 22.28555630001938,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 7,
      "seconds": 22.012718999991193,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 8,
      "seconds": 21.900937099999283,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 9,
      "seconds": 21.997255099995527,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 10,
      "seconds": 22.043873999995412,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 11,
      "seconds": 21.91684879999957,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 12,
      "seconds": 21.976963399996748,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 13,
      "seconds": 21.87815010000486,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 14,
      "seconds": 21.91939110000385,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 15,
      "seconds": 22.24360980000347,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 16,
      "seconds": 21.813998900004663,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 17,
      "seconds": 22.15224200001103,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 18,
      "seconds": 21.624001000018325,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    }
  ],
  "finalize_seconds": 0.6423304000054486,
  "scope": "two independent heads sequentially update in one process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate"
}
清理5个无亮点/冗余状态，共12834827字节；正源和交付不改。
未运行其他折或旧队列。原第五严格>80.5且明确收益后才考虑其余四折；完整原OOF>82未完成。目标后独立井验证和新目录交付。


## 收尾与图谱
两项均18轮patience18、恢复第0轮原函数；训练后joint比likelihood低0.000746818167，没有绝对或匹配效用增益。源/评分/预处理/冻结/最佳恢复与实际三列评分已通过，不重复旧Q检查。删除5个冗余状态，共12834827字节；原POR/PERM/SW正模型和交付保留。图谱630节点、86个dead end。一个GPU进程共享提取两臂总墙钟约413.26秒，不伪造单臂速度；GPU中途全卡5953MiB快照和PyTorch峰值752.12MiB定义分开，见artifact_metadata。没有其他GPU训练或补折。

目标：纯DL原完整OOF>82；当前仅固定第五。
已做：两项真实训练、匹配对照、最佳恢复、实际组合、必要重载与清理。
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


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.

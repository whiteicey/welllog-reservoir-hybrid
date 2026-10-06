# v88 PERM条件概率格读出最终结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|PERM|固定POR/SW TOTAL|最佳/停止轮|训练后最高PERM|
|---|---:|---:|---:|---:|
|likelihood|89.396724049366|80.382093291214|0/18|89.393663992748|
|joint|89.396724049366|80.382093291214|0/18|89.392037829129|

## 探索与验证
同37185小格头参数、同初始原函数，原PERM私有预测器冻结。log质量界0.5/gate界0.25，先验零格不能新增质量；原9曲线206+已有幅值63/分折/Q50/256支持800动作不改。标签为本目标训练log10双邻格软CE，效用0/1、温度.05；新head1e-4/cos至1e-5/80上限pat18。
实际第二batch效用梯度非零，来源/预处理/source hash、冻结状态、最佳恢复、144665点原行序和三列评价通过；正候选磁盘重建=[]。

## 收益与不继续
joint-likelihood含初始最终PERM +0.000000000000；训练后最高差 -0.001626163619。采用=None，保留=[]。epoch0恢复不算新收益，更新后最高为同容量完成对照，不替代当前模型选择；若正增益，两组更新后head/pred必须磁盘重载。
likelihood 默认/数值/井变化：{"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.23292459955995, "channel_delta": -0.011067361567739908, "TOTAL_contribution": -0.002558009113661313}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.1508877257431, "channel_delta": 0.012509467598022184, "TOTAL_contribution": 0.0014869892974051018}}；4胜/12负。
joint 默认/数值/井变化：{"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.25533636745075, "channel_delta": 0.011344406323061662, "TOTAL_contribution": 0.0026220427141421037}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.10252184644087, "channel_delta": -0.035856411704216384, "TOTAL_contribution": -0.00426221979710397}}；4胜/12负。
无绝对或完整训练后匹配收益，关闭当前冻结先验/有界乘性格读出/格CE或联合效用组合，不泛化全部概率格学习；单开发折选择非独立泛化确认。

## 成本、清理、未完成与目标后
{
  "shared_process_seconds": 121.42094959999667,
  "launch_to_last_completion_seconds": 126.016038,
  "launch_to_verified_seconds": 424.712068,
  "peak_training_processes": 1,
  "peak_allocated_MiB": 1031.92529296875,
  "epoch_timing": [
    {
      "epoch": 1,
      "seconds": 6.7000836999795865,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 2,
      "seconds": 6.0757811999937985,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 3,
      "seconds": 6.01797269997769,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 4,
      "seconds": 6.1781277999980375,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 5,
      "seconds": 5.975066400016658,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 6,
      "seconds": 6.011244900000747,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 7,
      "seconds": 6.070113099995069,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 8,
      "seconds": 6.008370900002774,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 9,
      "seconds": 6.0473033000016585,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 10,
      "seconds": 6.138328800007002,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 11,
      "seconds": 6.1127410999906715,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 12,
      "seconds": 6.054618499998469,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 13,
      "seconds": 5.977390700019896,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 14,
      "seconds": 6.138667899998836,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 15,
      "seconds": 5.954086899990216,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 16,
      "seconds": 6.005036899994593,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 17,
      "seconds": 5.94846729998244,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    },
    {
      "epoch": 18,
      "seconds": 5.90980860000127,
      "active_modes": [
        "likelihood",
        "joint"
      ]
    }
  ],
  "finalize_seconds": 0.5915643000043929,
  "scope": "two independent37185heads sequentially update in one GPU process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate"
}
清理5个无亮点/冗余状态，共8644644字节；正源和交付不改。
未运行其他折或旧队列。原第五严格>80.5且明确收益后才考虑其余四折；完整原OOF>82未完成。目标后独立井验证和新目录交付。

## 收尾
实际三列TOTAL80.382093291214不变；两项18轮patience18、恢复第0轮，训练后最佳似然89.393663992748/联合89.392037829129，匹配差-0.001626163619；不继续本乘性格读出或扫描界/温度。已清理5个冗余权重，共8644644字节，源/config/日志/成绩保留，原正源/交付不改。图谱635节点、88个dead end；无GPU训练、旧队列或其他折。训练两臂共享冻结计算总墙钟126.016秒，峰值allocated1031.925MiB/reserved1200MiB，准备和收尾另计。

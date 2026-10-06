# v85 SW最终残差输出结构结果

当前原第五固定三通道TOTAL 80.382093291214 → 80.382093291214。

|模型|新增参数|SW|固定POR/PERM组合TOTAL|
|---|---:|---:|---:|
|linear|0|73.793310952488|80.382093291214|
|point64|9881|73.750133719096|80.366981259527|
|local64|10137|73.747292935820|80.365986985381|

## 探索与验证
同原v56训练折内骨干/FiLM和SW先验重置，9881/10137新增输出参数，全网络更新。CPU fork_rng保护原随机态、共同64隐藏层初始化，零投影和局部中心恒等保证初始函数同历史线性头；基本状态hash和144665点初始预测同历史源。两者容量差256，近容量而非完全同参数，不将对照差完全归因于邻域。原输入/Q50/评分/解码/全人口效用1/40%DropKey/1e-4/80-18保持，不叠加人口重权或固定默认。
主body/FiLM/output和新hidden/project/局部卷积真实梯度、源码/评分/预处理/来源哈希、最佳恢复及实际三列评价通过；没有独立数值预检或审查。正候选原始输入磁盘重建：[]
point64最佳/停止14/32，对历史线性SW-0.043177233392/TOTAL-0.015112031687，默认+0.097787595562/数值-0.366960392785，井胜/负7/9。
local64最佳/停止14/32，对历史线性SW-0.046018016668/TOTAL-0.016106305834，默认+0.088983354133/数值-0.356103664873，井胜/负7/9。

## 收益与不继续
local-point SW-0.002840783276，训练后最高差-0.002840783276；绝对和结构比较分开，采用=None，保留=[]。
若两项都无绝对或预定义结构收益，关闭这两个小读出，不无界扩头或临时扫参数。开发折选择非独立泛化证明，容量差和实际停止轮如实记录；不是完整OOF或已超最终目标。

## 成本、清理与未完成
{
  "candidates": [
    {
      "mode": "point64",
      "start": "2026-09-30T11:51:25.763992+08:00",
      "end": "2026-09-30T11:57:53.810032+08:00",
      "seconds": 383.89757040000404,
      "peak_allocated_MiB": 1903.482421875,
      "launch_to_completion_seconds": 388.04604
    },
    {
      "mode": "local64",
      "start": "2026-09-30T11:52:20.752734+08:00",
      "end": "2026-09-30T11:58:49.251243+08:00",
      "seconds": 383.703964299988,
      "peak_allocated_MiB": 1911.8310546875,
      "launch_to_completion_seconds": 388.498509
    }
  ],
  "first_launch_to_last_completion_seconds": 443.487251,
  "first_launch_to_verified_wall_seconds": 757.801444,
  "peak_process_concurrency": 2,
  "process_overlap_seconds": 333.057298,
  "sum_individual_launch_to_completion_seconds": 776.544549,
  "finalization_seconds": 0.5896377000026405,
  "scope": "loading/preprocessing/warmup/training/evaluation/persistence and explicit launch gap included; authoring before first launch and atlas after score verification separate; observed admission may require serial runs; no throughput benchmark"
}
删除4无新增收益权重，共38525052字节；原正模型/来源/交付保留。
无其他折/旧队列；第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后独立井确认与新目录交付。

## 图谱与收尾
图谱已更新627个实验节点、85个dead end；新节点显示各自实际第五折组合分数80.366981259527/80.365986985381，不以采用模型80.382093替代负结果。训练进程已全部退出，来源和交付保留。负权重删除4个，共38525052字节；清单可核验。转向POR原有效路由的似然/联合效用有限对照，尚未启动。

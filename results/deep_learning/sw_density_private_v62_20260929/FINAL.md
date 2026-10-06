# v62密度私有骨干结果

固定第五折TOTAL 80.354325927816 → 80.354325927816。

|模式|训练后最高SW|含回退SW|最佳/停止轮|
|---|---:|---:|---|
|gate_aux|73.655212322576|73.713975628494|0/18|
|numeric_only|73.641491314940|73.713975628494|0/18|

## 探索重点与完成验证
同1678962总参数/839481优化器参数和完整初始函数；固定v56默认分支，另一个完整密度骨干可训练，推理用固定gate和私有密度。gate_aux用密度自己的gate作辅助监督，numeric_only系数0且该129参数训练无梯度、推理不用。其余原Q50/1024随机上下文/Gaussian损失/官方评分不变。
完整初始状态/输出头/144665点预测一致，私有骨干/头/FiLM实际梯度有效；固定分支权重及无梯度、初始/最佳恢复、源哈希和实际三列评分核验。新采用模型磁盘重建=False。

## 收益与方向边界
numeric-aux含回退SW差+0.000000000000；训练后最佳差-0.013721007637。采用=None。
第0轮回退不算训练收益。若仅数值监督无匹配收益，不扫描本辅助系数或支路数；结论限当前固定默认/私有密度结构，不否定全部任务解耦。正差仍为重复开发折证据，未作独立新井确认。

## 耗时与未做
{
  "candidates": [
    {
      "mode": "gate_aux",
      "start": "2026-09-29T21:45:58.152390+08:00",
      "end": "2026-09-29T21:51:21.362486+08:00",
      "seconds": 188.23036220000358,
      "peak_allocated_MiB": 1282.6796875,
      "initial_failure_at": "2026-09-29T21:47:53.754467+08:00",
      "failure": "CUDA OOM",
      "resumed_from_completed_epoch": 7,
      "attempts": [
        {
          "start": "2026-09-29T21:45:58.152390+08:00",
          "end": "2026-09-29T21:47:53.754467+08:00",
          "outcome": "failed_CUDA_OOM"
        },
        {
          "start": "2026-09-29T21:49:48.451649+08:00",
          "end": "2026-09-29T21:51:21.362486+08:00",
          "outcome": "completed_serial_resume"
        }
      ],
      "seconds_scope": "saved completed-epoch elapsed plus resumed execution; failed in-flight epoch work not included, see attempt wall times"
    },
    {
      "mode": "numeric_only",
      "start": "2026-09-29T21:46:03.195833+08:00",
      "end": "2026-09-29T21:53:25.652449+08:00",
      "seconds": 186.62775769998552,
      "peak_allocated_MiB": 1282.6796875,
      "initial_failure_at": "2026-09-29T21:47:53.770103+08:00",
      "failure": "CUDA OOM",
      "resumed_from_completed_epoch": 7,
      "attempts": [
        {
          "start": "2026-09-29T21:46:03.195833+08:00",
          "end": "2026-09-29T21:47:53.770103+08:00",
          "outcome": "failed_CUDA_OOM"
        },
        {
          "start": "2026-09-29T21:51:52.791342+08:00",
          "end": "2026-09-29T21:53:25.652449+08:00",
          "outcome": "completed_serial_resume"
        }
      ],
      "seconds_scope": "saved completed-epoch elapsed plus resumed execution; failed in-flight epoch work not included, see attempt wall times"
    }
  ],
  "first_launch_to_last_completion_seconds": 447.500059,
  "peak_process_concurrency": 2,
  "finalization_seconds": 0.32607629999984056,
  "scope": "initial concurrent attempts failed at epoch8 with CUDA OOM; serial explicit resume from epoch7; wall includes failed attempts, recovery gap, loading and warmup; finalization separate"
}
首次两项并发在第8轮CUDA OOM，保留failure.json与原日志；各自完整状态在第7轮，从原子模型/优化器/调度/RNG状态显式串行续跑。源码和配置未改，不是从头重训；失败不算负结论或成功。CUDA反向仍不保证逐位重训。完整五折OOF>82未完成，无其他折/交付修改。


## v62最终结果与保留边界
gate_aux：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
numeric_only：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
仅数值-辅助SW=+0.000000000000；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
仅数值监督无匹配收益，停止本辅助系数和支路数扫描；只限当前固定默认/私有密度结构，不泛化全部任务解耦。
同1678962总/839481优化器参数、完整初始状态及144665点预测相同；默认分支冻结，私有密度骨干全训练，自己的gate仅作可选辅助，numeric_only额外129个gate参数无梯度且不用于推理。原SW/Q/Gaussian损失、lr3e-5保持。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[188.23, 186.628]秒；初始并发失败加串行恢复的总墙钟447.500秒，包含失败尝试/恢复间隔/加载/输入库/预热；报告内seconds不含未保存的失败轮工作，attempts记录其墙钟；每任务峰值MiB=[1282.68, 1282.68]，代码与最终重建另计。
已删除4个本轮负检查点，共58450492字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。

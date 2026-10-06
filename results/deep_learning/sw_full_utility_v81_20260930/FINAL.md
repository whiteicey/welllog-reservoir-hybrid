# v81 SW全网络联合效用结果

当前独立通道路线第五折TOTAL 80.363582620846 → 80.380136601599。

|模式|重置先验SW|训练后最高SW|含起点最佳SW|最佳/停止轮|
|---|---:|---:|---:|---|
|likelihood|71.197115081472|73.735774494628|73.735774494628|25/43|
|joint|71.197115081472|73.787720410731|73.787720410731|25/43|

## 探索与验证
同v56迁移骨干/FiLM与重置SW先验，全839481参数训练，40%DropKey保持，原1e-4/80cosine。两组共同似然和可微效用图，辅助系数0/.5。原输入/数据/Q50/推理解码不改；两项与历史p40初始参数/全点预测一致，预算等23项匹配。源码/评分kernel/训练侧归一化哈希、四组真实网络梯度及辅助门控/密度路径、最佳恢复和三列评分通过；不单独预检。
保留正候选的原始输入磁盘重建：[{'mode': 'joint', 'raw_disk_replay_exact': True, 'rows': 144665, 'checkpoint_sha256': '8afbd80039ed0dec7f5fb09fdf4ef3a8653c7dd79028815eafe68fa4bdb38e1a'}]

## 实际收益与不继续
joint-likelihood 含起点差+0.051945916102，训练后最高差+0.051945916102；绝对和机制收益分别报告，采用=joint，保留=['joint']。
配对子群默认+0.051820436092，数值+0.052234132069，井胜/负11/5；开发折选择不当独立泛化证明。
若joint不优于同批likelihood，不扫描本完整DropKey配方的辅助温度/系数；仅限此同起点/预算/代理，不泛化全部效用学习。共同训练收益不能归给辅助项，平滑策略不等于官方argmax。

## 时间、清理与未完成
{
  "candidates": [
    {
      "mode": "likelihood",
      "start": "2026-09-30T10:15:10.912282+08:00",
      "end": "2026-09-30T10:21:19.054224+08:00",
      "seconds": 363.8191786999814,
      "peak_allocated_MiB": 1870.50390625,
      "launch_to_completion_seconds": 368.141942
    },
    {
      "mode": "joint",
      "start": "2026-09-30T10:18:13.559182+08:00",
      "end": "2026-09-30T10:24:17.496890+08:00",
      "seconds": 359.39152939998894,
      "peak_allocated_MiB": 1870.50390625,
      "launch_to_completion_seconds": 363.937708
    }
  ],
  "first_launch_to_last_completion_seconds": 546.584608,
  "first_launch_to_verified_wall_seconds": 579.349273,
  "last_training_completion_to_verified_wall_seconds": 32.764665,
  "peak_process_concurrency": 2,
  "process_overlap_seconds": 185.495042,
  "sum_individual_launch_to_completion_seconds": 732.0796499999999,
  "finalization_seconds": 2.6205762999888975,
  "scope": "individual seconds: trainer loading/preprocessing/warmup/training/evaluation/persistence; wall additionally includes process start and launch gap; authoring before launch and atlas after verification separate; no serial baseline, so no concurrency speedup claim"
}
删除2无新增收益/重复父模型checkpoint，共19051078字节；清单先写后删，原v77和历史正候选/交付保留。
max80/pat18/min10，实际最佳/停止轮见表。无其他折/旧队列，严格第五>80.5及完整OOF>82目标仍按实际结果判定，未完成不宣称达成。目标后独立井确认与新目录交付。

## v81图谱与最终留存核验
图谱节点显式显示第五折固定POR/PERM组合TOTAL80.380136601599及单目标SW73.787720410731；不混为完整OOF。离线生成618实验/82关闭路线，实际分数读回一致。联合组best/last与旧p40仍在，两个负对照checkpoint已删除19051078字节，v81训练进程为0。训练到验证579.349273秒；到本批文档/图谱收尾墙钟890.723秒，验证后收尾311.373秒（含定位组合字段、图谱更新和读回），详见artifact_finalize.json。无自动后续启动或补折；目标仍未完成。

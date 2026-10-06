# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v62 SW密度私有骨干与状态辅助监督

## 探索重点与依据
当前v56 SW73.713975628494/TOTAL80.354325927816；默认样本100786点官方SW97.883627，数值43879点18.198526。该验证分区只用于提出假设，不作逐行选择、训练权重或条件oracle。v59/v60冻结RQS头、v61插值均未改善，改为完整密度骨干与默认状态预测分离。
从v56第11轮复制两套完整SW网络：fixed全部冻结提供默认logit；density全骨干/FiLM/输出可训练，只提供连续密度。两组推理结构及完整初始参数相同，总1678962参数，其中839481在优化器。gate_aux对density自己的默认头加训练BCE；numeric_only系数0，仅数值密度梯度进入私有骨干。实际推理始终使用fixed的默认头，不用density的辅助默认预测。
两组损失=原Q加权[fixed默认BCE(常数)+非defaultGaussian NLL]/3 + alpha*原Q加权density默认BCE/3，alpha=1或0。保留常数项使日志与原NLL实现一致，它不产生梯度。numeric_only的129个辅助gate输出参数不接收梯度且不用于推理；名义容量相同，监督路径是变量。推理使用的密度分支参数为839352；不宣称两种目标梯度等价。
这不是旧模型投票：一个固定默认概率和一个私有条件密度组成单一SW分布，不读其他通道预测或验证标签。连续分布会影响最终动作，因此固定gate也不保证默认样本得分不变。v11曾在小型随机TCN上比较共享/分离，本轮是新冠军完整热启动、固定已验证gate、只改密度骨干辅助监督，不能将不同设置混同。

## 数据、执行与已实现
原9曲线/206、64训练井/16第五验证井144665点、Q50/seed20261010、1024随机锚点中心128、原8Gaussian/256支持点/800动作+默认评分保持。POR/PERM固定。训练标签仅原训练SW；整井bank只含输入，验证井仅自身推理，不使用标签拟合。
新AdamW3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps；两项显式启动，最多2GPU，无旧队列/其他折/交付覆盖。完整原子last_state含当前/最佳权重、优化器/调度/RNG/早停，恢复需显式。初始起点纳入选择，不算新训练收益。
已实现固定默认分支和私有密度分支，启动前无成绩。直接训练，真实批次检查固定分支无梯度、auxgate梯度与模式一致、私有骨干/头/FiLM有效；结尾固定权重哈希、源哈希、初始/最佳144665点及三列评分验证，不追加冗余预检或独立审查。采用或保留的模型需原始输入磁盘重建。

## 收益、待做和不继续
收益尚未证实。待两项训练、numeric_only-gate_aux配对差和相对当前分、记录参数/时间/显存、保留验证、清理负权重、PROJECTS/EXPLORATIONS/atlas。
若numeric_only无最终配对收益，不扫描当前辅助监督系数或复制支路数量；结论限该SW私有密度配置，不否定所有任务解耦。若两项都有更差结果则关闭此配置；不把起点回退当成功。重复开

希、初始/最佳144665点及三列评分验证，不追加冗余预检或独立审查。采用或保留的模型需原始输入磁盘重建。

## 收益、待做和不继续
收益尚未证实。待两项训练、numeric_only-gate_aux配对差和相对当前分、记录参数/时间/显存、保留验证、清理负权重、PROJECTS/EXPLORATIONS/atlas。
若numeric_only无最终配对收益，不扫描当前辅助监督系数或复制支路数量；结论限该SW私有密度配置，不否定所有任务解耦。若两项都有更差结果则关闭此配置；不把起点回退当成功。重复开发折差未经过独立井确认。
单项与两任务墙钟分开，记录加载/输入库/预热/训练/评估和并发，代码与最终重建另列。第五严格>80.5及完整原五折OOF>82仍未完成；未经门槛与明确收益不补折，旧交付不改。目标完成后可独立确认并新目录交付。


## v62 gate_aux explicit serial recovery
{"pid": 55664, "started_at": "2026-09-29T21:49:48.451649+08:00", "mode": "gate_aux", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "gate_aux", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_gate_aux_fold5", "--resume"], "serial_recovery": true, "automatic_queue": false, "reason": "Both initial jobs failed with CUDA OOM after completed epoch7; source/config unchanged; resume atomic optimizer/scheduler/RNG state"}


## v62 numeric_only explicit serial recovery
{"pid": 46332, "started_at": "2026-09-29T21:51:52.791342+08:00", "mode": "numeric_only", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "numeric_only", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_numeric_only_fold5", "--resume"], "serial_recovery": true, "automatic_queue": false, "reason": "Both initial jobs failed with CUDA OOM after completed epoch7; source/config unchanged; resume atomic optimizer/scheduler/RNG state"}


## v62密度私有骨干对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后numeric-aux=-0.013721007637。详见sw_density_private_v62_20260929/FINAL.md；无其他折或交付修改。


## v62最终结果与保留边界
gate_aux：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
numeric_only：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
仅数值-辅助SW=+0.000000000000；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
仅数值监督无匹配收益，停止本辅助系数和支路数扫描；只限当前固定默认/私有密度结构，不泛化全部任务解耦。
同1678962总/839481优化器参数、完整初始状态及144665点预测相同；默认分支冻结，私有密度骨干全训练，自己的gate仅作可选辅助，numeric_only额外129个gate参数无梯度且不用于推理。原SW/Q/Gaussian损失、lr3e-5保持。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[188.23, 186.628]秒；初始并发失败加串行恢复的总墙钟447.500秒，包含失败尝试/恢复间隔/加载/输入库/预热；报告内seconds不含未保存的失败轮工作，attempts记录其墙钟；每任务峰值MiB=[1282.68, 1282.68]，代码与最终重建另计。
已删除4个本轮负检查点，共58450492字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## FINAL.md

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
      "seconds_sc

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



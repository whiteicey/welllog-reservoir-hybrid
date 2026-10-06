# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# 三通道独立模型：全骨干单任务优化

## 目标与新授权
用户明确允许三个通道采用不同模型、分别训练，不要求共享backbone。本批解除此前只训POR/冻结骨干的限制：POR、PERM、SW分别拥有独立实例、可训练特征提取器、门控、密度头、优化器、停止计数和checkpoint。允许最终选择不同结构，旧实验和已交付版本保持不变。

## 六个预定义候选
对每个目标分别比较：dual保留完整206维视图及校准68维视图两条私有多尺度U-Net+局部/全局注意力骨干，并训练私有输入路由；single仅保留完整206维视图的一条私有骨干，不使用第二专家或路由。共3目标×2结构，不再扫描宽度/种子。三个目标的模型没有共享Parameter、优化器或可复用的训练中隐藏缓存。
初始化来自同一训练折的既有多任务预训练权重和v5读出。这是独立单任务微调，不声称全流程从零训练时也完全无多任务预训练。dual初始应逐元素复现旧v5对应通道；single起点是原第一专家，需单独报告，不能冒称和dual初始函数相同。

## 训练协议
所有有效特征提取参数都能更新，包括stem、编码/解码、局部/全局attention；dual的校准和路由也更新。只有当前目标的门控和密度输出行可训练，其他输出行保留为冻结buffer以匹配原卷积计算，不计算其他目标的损失。只需一个目标的预测分布；无跨目标梯度、无外部B/D融合。
骨干/路由lr1e-5，目标头lr3e-5，AdamW、wd0.01、余弦80×64步至1e-6、clip2。原batch64、64步/轮、max80/patience18/min10及seed20261010；按自身通道官方分选最佳epoch（允许0）。骨干训练时启用原dropout，所有模型使用各自独立训练状态。
原九曲线/206特征、原64训练井16验证井、578891训练点144665验证点、Q50和单覆盖256格800动作保持。共享相同的确定性预处理定义不等于共享可学习骨干。每项目标用自身BCE+非默认NLL，POR Student-t3，PERM/SW Gaussian；除3保留历史loss尺度。标签监督只读当前通道。

## 收益与验证
启动前没有结果。每项报告自己的通道分，以及将该通道替换进旧参考时的诊断总分；后者不得当完整独立三模型成绩。六项完成后为每个目标固定一种模型，三份完整checkpoint真实重载推理，合并同一144665点再计算官方TOTAL。
正式训练检查初始、骨干梯度、实际骨干权重变化、最终最佳恢复和冻结packed行；不另开GPU预检任务。保存全模型current/best、optimizer/scheduler/RNG/停止状态，只有显式resume。首次训练真实错误单独记录，旧源码不改。

## 边界与成本
最多2GPU任务，显式分批启动，不设置自动队列；本批最多六项。未超过80.5也可记录单通道结构收益，但不补折；完整OOF>82仍未完成。单任务训练相对共享模型也改变了更新范围和总预算，因此不能将差异都归因为消除梯度冲突。不同结构/目标独立选型属于开发选择，需保留跨折确认限制。
最终3个模型需要3到6条骨干，计算和储存可能高于原共享两骨干；不把精度进展与推理成本混为一谈。无新数据、测试标签或旧ABD恢复，交付不覆盖。每批

# 三通道独立模型：全骨干单任务优化

## 目标与新授权
用户明确允许三个通道采用不同模型、分别训练，不要求共享backbone。本批解除此前只训POR/冻结骨干的限制：POR、PERM、SW分别拥有独立实例、可训练特征提取器、门控、密度头、优化器、停止计数和checkpoint。允许最终选择不同结构，旧实验和已交付版本保持不变。

## 六个预定义候选
对每个目标分别比较：dual保留完整206维视图及校准68维视图两条私有多尺度U-Net+局部/全局注意力骨干，并训练私有输入路由；single仅保留完整206维视图的一条私有骨干，不使用第二专家或路由。共3目标×2结构，不再扫描宽度/种子。三个目标的模型没有共享Parameter、优化器或可复用的训练中隐藏缓存。
初始化来自同一训练折的既有多任务预训练权重和v5读出。这是独立单任务微调，不声称全流程从零训练时也完全无多任务预训练。dual初始应逐元素复现旧v5对应通道；single起点是原第一专家，需单独报告，不能冒称和dual初始函数相同。

## 训练协议
所有有效特征提取参数都能更新，包括stem、编码/解码、局部/全局attention；dual的校准和路由也更新。只有当前目标的门控和密度输出行可训练，其他输出行保留为冻结buffer以匹配原卷积计算，不计算其他目标的损失。只需一个目标的预测分布；无跨目标梯度、无外部B/D融合。
骨干/路由lr1e-5，目标头lr3e-5，AdamW、wd0.01、余弦80×64步至1e-6、clip2。原batch64、64步/轮、max80/patience18/min10及seed20261010；按自身通道官方分选最佳epoch（允许0）。骨干训练时启用原dropout，所有模型使用各自独立训练状态。
原九曲线/206特征、原64训练井16验证井、578891训练点144665验证点、Q50和单覆盖256格800动作保持。共享相同的确定性预处理定义不等于共享可学习骨干。每项目标用自身BCE+非默认NLL，POR Student-t3，PERM/SW Gaussian；除3保留历史loss尺度。标签监督只读当前通道。

## 收益与验证
启动前没有结果。每项报告自己的通道分，以及将该通道替换进旧参考时的诊断总分；后者不得当完整独立三模型成绩。六项完成后为每个目标固定一种模型，三份完整checkpoint真实重载推理，合并同一144665点再计算官方TOTAL。
正式训练检查初始、骨干梯度、实际骨干权重变化、最终最佳恢复和冻结packed行；不另开GPU预检任务。保存全模型current/best、optimizer/scheduler/RNG/停止状态，只有显式resume。首次训练真实错误单独记录，旧源码不改。

## 边界与成本
最多2GPU任务，显式分批启动，不设置自动队列；本批最多六项。未超过80.5也可记录单通道结构收益，但不补折；完整OOF>82仍未完成。单任务训练相对共享模型也改变了更新范围和总预算，因此不能将差异都归因为消除梯度冲突。不同结构/目标独立选型属于开发选择，需保留跨折确认限制。
最终3个模型需要3到6条骨干，计算和储存可能高于原共享两骨干；不把精度进展与推理成本混为一谈。无新数据、测试标签或旧ABD恢复，交付不覆盖。每批维护PROJECTS与演进图谱。


## CUDA反向兼容性修复
首批两模型初始目标预测逐元素复现，但严格确定性在首次linear插值反向报错，未执行optimizer.step。失败attempt2完整保存在各目录failed_attempt2。六项统一采用deterministic warn_only保留可用确定性算子，对CUDA插值反向允许警告；固定随机种子仍不能保证训练逐位重现。该设置已写入config。旧源代码未改，明确从原起点重新启动首批，不冒称续跑。


## FINAL.md

# 三通道独立模型：实际结果

2026-09-28T12:02:48+08:00

|目标|结构|状态|通道分|相对旧参考|最佳/停止轮次|
|---|---|---|---:|---:|---|
|POR|dual|完成|77.448915708|+0.000000000|0/18|
|POR|single|完成|77.034676101|-0.414239606|0/18|
|PERM|dual|完成|89.347502734|+0.000000000|0/18|
|PERM|single|完成|89.120208951|-0.227293783|0/18|
|SW|dual|完成|73.622384767|+0.000000000|0/18|
|SW|single|完成|73.100736795|-0.521647972|0/18|

## 当前选择（只使用已完成候选）
[
  {
    "target": "POR",
    "key": "POR_dual",
    "target_score": 77.44891570750882,
    "delta_reference": 0.0
  },
  {
    "target": "PERM",
    "key": "PERM_dual",
    "target_score": 89.3475027344081,
    "delta_reference": 0.0
  },
  {
    "target": "SW",
    "key": "SW_dual",
    "target_score": 73.62238476712884,
    "delta_reference": 0.0
  }
]

六项全部完成=True；独立三模型共同重载=True。
每个目标只用自身监督、私有模型与优化器。初始化来自既有多任务预训练；当前骨干允许更新。不同目标不共享live参数或训练中间特征。
单通道替换参考TOTAL只用于诊断；只有固定三模型真实预测合并后才报告本轮完整TOTAL。
single为一条完整特征骨干，dual为完整+校准视图两条私有骨干，各通道允许选择不同结构。没有额外宽度/LR搜索。
独立模型相对旧共享模型也增加了更新范围和总训练预算；不能把总收益单独归因为梯度冲突消除。
初次POR/SW启动漏sys导入、未进入训练；已修复并保留首轮失败日志。其后真实状态以各run记录为准。
保持原9曲线/206特征、原分折/144665评分人口、Q和单覆盖解码；不自动补折，不覆盖旧交付。完整OOF>82尚未达成。

## 真实三模型结果
{
  "fold": 5,
  "screening_fold": 5,
  "scores": {
    "POR": 77.44891570750882,
    "PERM": 89.3475027344081,
    "SW": 73.62238476712884,
    "TOTAL": 80.27413533779057
  },
  "reference_scores": {
    "POR": 77.44891570750882,
    "PERM": 89.3475027344081,
    "SW": 73

8,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 37,
      "n": 9939,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 50,
      "n": 11874,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 54,
      "n": 9630,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 63,
      "n": 5639,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 69,
      "n": 9120,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 71,
      "n": 4709,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 72,
      "n": 9559,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 76,
      "n": 7109,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 79,
      "n": 10629,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    }
  ],
  "exploratory_selection_unadjusted_95": [
    0.0,
    0.0
  ],
  "timing": [
    {
      "target": "POR",
      "seconds_full_validation": 3.0552985000540502
    },
    {
      "target": "PERM",
      "seconds_full_validation": 1.5426295999786817
    },
    {
      "target": "SW",
      "seconds_full_validation": 1.5606318000354804
    }
  ],
  "setup_and_three_replays_seconds": 55.92976440000348,
  "score_scope": "actual three independent target models on original full fifth fold",
  "complete_five_folds": false,
  "remaining_folds_started": false,
  "screening_threshold": 80.5,
  "screening_passed": false,
  "development_selection": true,
  "new_shared_training_control_run": false,
  "limitations": "Repeated fold5selection and pretrained multitarget initialization; no independent test. Update scope and total compute differ from old frozen model.",
  "finished_at": "2026-09-28T11:48:13.776120+08:00"
}



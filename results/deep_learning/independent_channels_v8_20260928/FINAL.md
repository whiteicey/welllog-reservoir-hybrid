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
    "SW": 73.62238476712884,
    "TOTAL": 80.27413533779057
  },
  "delta_reference": 0.0,
  "choices": [
    {
      "target": "POR",
      "key": "POR_dual",
      "architecture": "dual",
      "target_score": 77.44891570750882,
      "best_epoch": 0,
      "checkpoint": "outputs/pure_dl_independent_v8_POR_dual80_fold5/model_best.pt"
    },
    {
      "target": "PERM",
      "key": "PERM_dual",
      "architecture": "dual",
      "target_score": 89.3475027344081,
      "best_epoch": 0,
      "checkpoint": "outputs/pure_dl_independent_v8_PERM_dual80_fold5/model_best.pt"
    },
    {
      "target": "SW",
      "key": "SW_dual",
      "architecture": "dual",
      "target_score": 73.62238476712884,
      "best_epoch": 0,
      "checkpoint": "outputs/pure_dl_independent_v8_SW_dual80_fold5/model_best.pt"
    }
  ],
  "three_distinct_full_model_checkpoints": true,
  "no_shared_parameter_storage": true,
  "all144665rows_reloaded_exactly": true,
  "original_row_order_verified": true,
  "private_backbone_branches": 6,
  "total_model_parameters": 3853902,
  "selected_backbone_changes": {
    "POR": 0.0,
    "PERM": 0.0,
    "SW": 0.0
  },
  "well_wins": 0,
  "well_losses": 0,
  "per_well": [
    {
      "well_index": 7,
      "n": 10219,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 8,
      "n": 9011,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 14,
      "n": 12078,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 16,
      "n": 8149,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 25,
      "n": 7739,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 34,
      "n": 10913,
      "TOTAL_delta": 0.0,
      "target_deltas": {
        "POR": 0.0,
        "PERM": 0.0,
        "SW": 0.0
      }
    },
    {
      "well_index": 35,
      "n": 8348,
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

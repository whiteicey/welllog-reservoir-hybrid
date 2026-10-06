"""Read-only native fusion analysis of completed folds; no ratio selection."""
import json,sys
from pathlib import Path
import numpy as np

study=Path(__file__).resolve().parent
root=study.parents[2]
sys.path.insert(0,str(root/'experiments/pure_dl'))
from fullcv_context_v97 import original_population
from train_invariant import metric,TS,dmask

population=original_population();completed=[]
for path in sorted(study.glob('fold_*/FOLD_REPORT.json')):
    report=json.loads(path.read_text(encoding='utf-8'));fold=report['fold']
    truth=population['truth'][population['fold']==fold-1];mask=dmask(truth)
    assert len(truth)==report['rows'] and report['native_replay_exact'] and report['prediction_disk_readback_exact']
    strata={}
    for label in ('original','seed13','fixed','diagnostic'):
        pred=np.load(path.parent/(label+'_prediction.npy'))
        assert pred.shape==truth.shape and np.isfinite(pred).all()
        expected=report['standalone_scores'][label] if label in ('original','seed13') else report[label+'_scores']
        actual=metric(truth,pred)
        assert all(abs(actual[k]-expected[k])<1e-9 for k in (*TS,'TOTAL'))
        strata[label]={}
        for c,target in enumerate(TS):
            strata[label][target]={}
            for name,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]:
                subset=truth[m];candidate=subset.copy();candidate[:,c]=pred[m,c]
                strata[label][target][name]=dict(rows=int(m.sum()),score=metric(subset,candidate)[target])
    gains={}
    for label in ('fixed','diagnostic'):
        gains[label]={target:{name:strata[label][target][name]['score']-strata['original'][target][name]['score']
            for name in ('DEFAULT','CONT')} for target in TS}
    analysis=dict(fold=fold,rows=len(truth),report=report,strata=strata,gains_against_original=gains,
        labels_used_for_analysis_only=True,rowwise_model_choice=False,weights_changed=False,
        scope='fifth-selected fixed ratios; foldwise best is tuning diagnostic, not independent fullOOF')
    (path.parent/'FUSION_STRATA.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    completed.append(analysis)
lines=['# 已完成折的两种子融合结果','',
    '固定比例为原模型 POR56% / PERM44% / SW68%，来自第五折选择，并原样用于其余折。逐头最优比例使用本折验证标签扫描，仅作调参诊断。两模型各8阶段、合计16个历史训练阶段；融合扫描不新增训练阶段。',
    '表内仅列已完成并通过原生磁盘重载、全行覆盖及预测写盘核验的折，未完成折不填分。完整五折OOF仍须等待全部723556行结果。','',
    '| 折 | 原模型 | 新种子 | 固定比例 | 逐头最优诊断 | POR原/新 | PERM原/新 | SW原/新 |',
    '|---:|---:|---:|---:|---:|---|---|---|']
for analysis in completed:
    r=analysis['report'];w=r['per_head_best_weights']
    ratios=[f"{100*w[t]['weight_original']:.0f}%/{100*w[t]['weight_seed13']:.0f}%" for t in TS]
    lines.append(f"| {r['fold']} | {r['standalone_scores']['original']['TOTAL']:.6f} | {r['standalone_scores']['seed13']['TOTAL']:.6f} | {r['fixed_scores']['TOTAL']:.6f} | {r['diagnostic_scores']['TOTAL']:.6f} | "+' | '.join(ratios)+' |')
lines+=['','各通道默认值/连续值增益详见每折 `FUSION_STRATA.json`。这只是分组分析，两类样本使用相同的预测文件和同一个头权重，没有按标签类别或逐行选择模型。','']
(study/'FOLD_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps([dict(fold=a['fold'],fixed_TOTAL=a['report']['fixed_scores']['TOTAL'],
    diagnostic_TOTAL=a['report']['diagnostic_scores']['TOTAL'],fixed_gains_by_stratum=a['gains_against_original']['fixed']) for a in completed]))

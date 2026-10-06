"""Summarize completed subset inference and independent training ablations."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/pure_dl_8040_pair_ablation'
LABELS={0:'A 原双专家seed06',2:'B 双专家seed08',3:'C 校准双专家75',4:'D 单骨干'}


def main():
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    arms={v['arm']:v for v in report['arms']}
    baseline=arms['four_fine__baseline4']
    costs=json.loads((OUT/'model_costs.json').read_text(encoding='utf-8'))
    cost_map={tuple(v['sources']):v for v in costs['subsets']}
    rng=np.random.default_rng(20261119)
    bootstrap=rng.integers(0,16,(10000,16))
    rows=[]
    for name,a in arms.items():
        short=name.removeprefix('four_fine__')
        if short=='baseline4':kept=[0,2,3,4]
        elif short.startswith('triple_drop_'):kept=[k for k in [0,2,3,4] if k!=int(short.rsplit('_',1)[1])]
        else:kept=list(map(int,short.split('_')[1:]))
        cost=cost_map[tuple(kept)]
        n=np.array([v['points'] for v in a['per_well']])
        assert [v['well'] for v in a['per_well']]==[v['well'] for v in baseline['per_well']]
        delta=np.array([x['scores']['TOTAL']-y['scores']['TOTAL'] for x,y in zip(a['per_well'],baseline['per_well'])])
        resampled=(n[bootstrap]*delta[bootstrap]).sum(1)/n[bootstrap].sum(1)
        rows.append(dict(arm=short,sources=kept,label='+'.join(LABELS[k].split()[0] for k in kept),
                         scores=a['scores'],delta=a['scores']['TOTAL']-baseline['scores']['TOTAL'],
                         parameters=cost['parameters'],backbones=cost['backbones'],wells_positive=int((delta>0).sum()),
                         exploratory_ci95=np.quantile(resampled,[.025,.975]).tolist()))
    rows.sort(key=lambda x:(-len(x['sources']),-x['scores']['TOTAL']))
    trained=[]
    control=json.loads((ROOT/'outputs/pure_dl_paired_posterior_control_compact80_sequence/fold_5_report.json').read_text())
    definitions=[('no_quality','关闭动态Q'),('gaussian_por','POR改Gaussian')]
    for mode,title in [('main_only','关闭辅助监督'),('no_calibration','关闭输入校准')]:
        if (ROOT/('outputs/pure_dl_8039_ablation_'+mode)/'fold_5_report.json').exists():
            definitions.append((mode,title))
    for mode,title in definitions:
        p=ROOT/('outputs/pure_dl_8039_ablation_'+mode)
        result=json.loads((p/'fold_5_report.json').read_text())
        assert result['best_checkpoint_predictions_reproduced'] and result['all_validation_rows_covered_once']
        trained.append(dict(mode=mode,title=title,scores=result['scores'],delta=result['scores']['TOTAL']-control['scores']['TOTAL'],
                            channel_delta={k:result['scores'][k]-control['scores'][k] for k in ['POR','PERM','SW']},
                            best_epoch=result['best_epoch'],stopped_epoch=result['stopped_epoch'],seconds=result['seconds'],
                            peak_gpu_mib=result['peak_gpu_allocated_mib'],checkpoint_persisted=result['checkpoint_persisted']))
    summary=dict(baseline=baseline['scores'],subsets=rows,training_control=control['scores'],training_ablations=trained,
                 reference_verification=report['reference_verification'],scope='Repeated fixed-fifth development ablations; no independent held-out or full OOF claim.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 80.4032基线：模型删减及训练消融结果','','![已完成消融对比](attribution_summary.png)','',
          '固定原第五折144665点。所有模型删减组共享相同四重余弦窗口、精细解码和等权分布融合；不重新训练、不拟合子集权重。',
          '', '；'.join(LABELS.values())+'。','',
          '| 组合 | 来源数/骨干数 | 参数量 | TOTAL | 相对四来源 | POR | PERM | SW |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in rows:
        s=v['scores'];text.append(f"|{v['label']}|{len(v['sources'])}/{v['backbones']}|{v['parameters']:,}|{s['TOTAL']:.6f}|{v['delta']:+.6f}|{s['POR']:.4f}|{s['PERM']:.4f}|{s['SW']:.4f}|")
    text+=['','逐井胜负及探索性井重采样区间见summary.json；16口验证井且已反复用于开发，区间不构成独立显著性证据。参数减少不等于实测提速。','',
           '## 独立训练消融','',f"对照是原双专家紧凑版本{control['scores']['TOTAL']:.6f}，不是四来源融合{baseline['scores']['TOTAL']:.6f}；不能跨对照相减归因。",'',
           '| 变更 | TOTAL | 相对原双专家 | 最佳/结束轮 |','|---|---:|---:|---:|']
    for v in trained:text.append(f"|{v['title']}|{v['scores']['TOTAL']:.6f}|{v['delta']:+.6f}|{v['best_epoch']}/{v['stopped_epoch']}|")
    text+=['','表中已完成训练均恢复最佳checkpoint并覆盖全部验证点，核验通过。Gaussian对照同步修改损失、解码和探测，测试的是分布族机制；Q关闭只改变最终预测损失权重。其他行（如有）分别只关闭辅助损失或可学习输入校准。单seed结果不能视为普遍因果证明，各项贡献不能直接相加。','',
           '## 复现和范围','',json.dumps(report['reference_verification'],ensure_ascii=False), '',
           '本报告不是完整五折结果，目标完整OOF>82尚未证实。原权重、交付文件不覆盖；最高组合来自当前验证诊断，需如实报告选择过程。']
    (OUT/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

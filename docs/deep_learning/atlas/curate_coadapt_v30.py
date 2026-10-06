"""Incremental v30 annotations; existing experimental evidence is preserved."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    study=root/'results/deep_learning/champion_coadapt_v30_20260929'
    result=json.loads((study/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_six_complete'] and result['actual_combined_prediction_scored']
    path=here/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    for s in result['selections']:
        t=s['target'];key='pure_dl_coadapt_v30_'+t+'_fold5'
        records=[r for r in result['candidates'] if r['target']==t]
        source='outputs/'+key+'/fold_5_report.json'
        detail='；'.join(f"{r['mode']}选定{r['report']['target_score']:.9f}@{r['report']['best_epoch']}，更新后{r['post_update']['scores'][t]:.9f}" for r in records)
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title='v30 '+t+'协同微调 · '+(s['mode'] or '保留v29'),stage='architecture',status='recorded',
            highlight=bool(s['adopted']),primary=dict(value=s['target_score'],scope='原第五折 · '+t+'单通道',kind='diagnostic',source=source),
            summary='同结构和初始函数，仅路由/路由加有效修正头共同更新；无新增模块。',
            conclusion=detail+f"。实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}。单一开发折结果，非独立泛化确认。",
            evidence=[source,'results/deep_learning/champion_coadapt_v30_20260929/assembled_report.json','results/deep_learning/champion_coadapt_v30_20260929/FINAL.md'])
    temp=path.with_suffix('.v30.tmp.json');temp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    marker='## v30最终图谱'
    note=marker+'\n'+datetime.now().astimezone().isoformat()+f"；保留六项实测结果、v29父关系及选定组合{result['selected_scores']['TOTAL']:.12f}。未增加结构/其他折/旧任务，旧模型保留。\n"
    p=here/'PROJECTS.md'
    if marker not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+note)
    print(json.dumps(dict(scores=result['selected_scores'],selections=[dict(target=s['target'],mode=s['mode']) for s in result['selections']]),ensure_ascii=False))


if __name__=='__main__':main()

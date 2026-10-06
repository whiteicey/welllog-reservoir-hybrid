"""Incrementally finalize v29 evidence while preserving historical curation."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    study=root/'results/deep_learning/champion_split_route_v29_20260929'
    result=json.loads((study/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_six_complete'] and result['actual_combined_prediction_scored']
    path=here/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    for selected in result['selections']:
        t=selected['target'];key='pure_dl_split_route_v29_'+t+'_fold5'
        rows=[r for r in result['candidates'] if r['target']==t]
        source='outputs/'+key+'/fold_5_report.json'
        text='；'.join(f"{r['mode']}={r['report']['target_score']:.9f}@{r['report']['best_epoch']}，停止{r['report']['stopped_epoch']}" for r in rows)
        node=cur.setdefault('nodes',{}).setdefault(key,{})
        node.update(title='v29 '+t+'路由分工 · '+(selected['mode'] or '保留v28'),stage='architecture',status='recorded',
            highlight=bool(selected['adopted']),
            primary=dict(value=selected['target_score'],scope='原第五折 · '+t+'单通道',kind='diagnostic',source=source),
            summary='冻结v28，耦合/分离路由增量各20546参数；初始函数不变，单项增量±0.125。',
            conclusion=text+f"。采用{selected['mode']}；实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}。开发折结果，未做独立泛化确认。",
            evidence=[source,'results/deep_learning/champion_split_route_v29_20260929/FINAL.md',
                'results/deep_learning/champion_split_route_v29_20260929/assembled_report.json'])
    temp=path.with_suffix('.v29.tmp.json');temp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    marker='## v29最终图谱'
    note=marker+'\n'+datetime.now().astimezone().isoformat()+f"；记录六项匹配结果及实际组合{result['selected_scores']['TOTAL']:.12f}，父模型v28保持。独立增量脚本保留历史curation，未重写原curate.py。\n"
    p=here/'PROJECTS.md'
    if marker not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as stream:stream.write('\n\n'+note)
    print(json.dumps(dict(scores=result['selected_scores'],selections=[dict(target=s['target'],mode=s['mode']) for s in result['selections']]),ensure_ascii=False))


if __name__=='__main__':main()

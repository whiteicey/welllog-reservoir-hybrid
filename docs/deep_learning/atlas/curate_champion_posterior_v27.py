"""Incremental v27 annotations; preserve every prior curated experiment."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    study=root/'results/deep_learning/champion_posterior_v27_20260929'
    data=json.loads((study/'progress.json').read_text(encoding='utf-8'))
    path=here/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    ids=[]
    for row in data['rows']:
        t=row['target'];key=row['id'];ids.append(key);rep=row['report'];state=row['state']
        source='outputs/'+key+('/fold_5_report.json' if rep else '/runtime_status.json')
        best=max(a['target_score'] for a in rep['arms']) if rep else max((a['best'] for a in state.get('modes',{}).values()),default=None)
        detail='；'.join(f"{a['mode']}更新后{a['best_after_update']['target_score']:.9f}，选定{a['target_score']:.9f}@{a['best_epoch']}，停止{a['stopped_epoch']}" for a in rep['arms']) if rep else '运行快照，尚无最终结果'
        cur.setdefault('nodes',{})[key]=dict(title='v27 '+t+'后验条件头',stage='architecture',status='recorded' if rep else state['status'],
            primary=dict(value=best,scope='原第五折 · '+t+'单通道（含第0轮）',kind='diagnostic',source=source) if best is not None else None,
            summary='保持80.300730612039父预测器；匹配self重复本专家摘要、pair读取另一专家摘要；只更新该目标已有效因子。',
            conclusion=detail+'。比较来源有限，正负结果不泛化；总分与实际选择见研究目录FINAL.md。',
            evidence=[source,'results/deep_learning/champion_posterior_v27_20260929/METHOD.md',
                'outputs/'+key+'/self/fold_5_report.json','outputs/'+key+'/pair/fold_5_report.json'])
        parent='pure_dl_champion_shape_v25_amplitude_fold5' if t=='POR' else 'pure_dl_all_heads_v26_'+t+'_fold5'
        edge=dict(source=parent,target=key,type='evolution',note='保留已采纳小头，冻结全部父参数，仅新增后验条件修正',evidence=source)
        if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):cur['edges'].append(edge)
    final=study/'assembled_report.json'
    if final.exists():
        res=json.loads(final.read_text(encoding='utf-8'))
        if not any(x['adopted'] for x in res['selections']):
            end=dict(id='dead:champion_posterior_v27_residual',title='v27重复追加后验条件头·本配置无增益',stage='architecture',parents=ids,
                summary='三个有效因子共六项self/pair均18轮停止并选择第0轮，最终仍80.300730612039。',
                reason='自身或配对后验摘要没有使追加的小头超过当前完整预测器。本轮不扩大该设置的宽度/损失/种子扫描，不否定其他后验建模。')
            cur['deadends']=[x for x in cur.get('deadends',[]) if x.get('id')!=end['id']]+[end]
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    note='\n\n## v27后验条件比较图谱\n'+datetime.now().astimezone().isoformat()+'；3个目标节点保留6个子实验分数与证据，单通道诊断不冒充TOTAL。调用curate_champion_posterior_v27.py增量维护；原curate.py编辑被工具内容保护拒绝，未修改。保留现有人工摘要，再运行build_atlas.py。\n'
    with (here/'PROJECTS.md').open('a',encoding='utf-8') as f:f.write(note)
    print('V27_CURATED',json.dumps(dict(all_complete=data['all_complete'],nodes=ids),ensure_ascii=False))


if __name__=='__main__':main()

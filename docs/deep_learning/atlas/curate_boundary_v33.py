"""Incremental v33 annotations; preserves all existing manually curated nodes."""
from pathlib import Path
from datetime import datetime
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STUDY=ROOT/'results/deep_learning/boundary_context_v33_20260929'

def main():
    path=HERE/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    refs=json.loads((ROOT/'results/deep_learning/mechanism_checks_v31_20260929/CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    final_path=STUDY/'assembled_report.json'
    final=json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else None
    for t in ('POR','SW'):
        key='pure_dl_boundary_v33_'+t+'_fold5';out=ROOT/'outputs'/key
        stat=out/'runtime_status.json'
        if not stat.exists():continue
        state=json.loads(stat.read_text(encoding='utf-8'));complete=state.get('status')=='complete'
        score=max([refs[t]]+[v['best'] for v in state.get('modes',{}).values()])
        desc='；'.join(m+' best='+str(v['best'])+'@'+str(v['best_epoch']) for m,v in state.get('modes',{}).items())
        source='outputs/'+key+('/fold_5_report.json' if complete else '/runtime_status.json')
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title='v33 '+t+'输入边界统计',stage='architecture',status='recorded' if complete else 'running',
            highlight=bool(final and score>refs[t]+1e-9),
            primary=dict(value=score,kind='diagnostic',scope='原第五折 · '+t+'单通道'+('' if complete else '运行中'),source=source),
            summary='固定原预测器；18维固定51点与输入边界截断统计，参数/初始化/缩放相同。',
            conclusion=desc+('。四项已完成，组合评价见研究报告。' if final else '。运行快照，尚未完成组合评价。'),
            evidence=[source,'results/deep_learning/boundary_context_v33_20260929/diagnostic.json','results/deep_learning/boundary_context_v33_20260929/feature_manifest.json'])
        parent='pure_dl_mechanism_v31_POR_fold5' if t=='POR' else 'pure_dl_split_route_v29_SW_fold5'
        edge=dict(source=parent,target=key,type='evolution',note='原完整模型冻结；只添加边界统计小头。',evidence=source)
        edges=cur.setdefault('edges',[])
        if not any(e.get('source')==parent and e.get('target')==key for e in edges):edges.append(edge)
        dead='dead:boundary_v33_'+t
        cur['deadends']=[x for x in cur.get('deadends',[]) if x.get('id')!=dead]
        if final and score<=refs[t]+1e-9:
            cur['deadends'].append(dict(id=dead,title='v33 '+t+'当前边界侧特征未获增益',stage='architecture',parents=[key],summary=desc,
                reason='输入边界困难证据成立，但本次截断51点统计与冻结头接入未超越父模型；停止该设置，不否定其他边界建模。'))
    tmp=path.with_suffix('.v33.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    marker='## v33 输入边界上下文图谱'
    p=HERE/'PROJECTS.md'
    if marker not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n'+datetime.now().astimezone().isoformat()+'；增量条目由curate_boundary_v33.py维护，原curate.py和其他节点不覆盖。最终分数口径为单通道第五折。\n')
    print('V33_ATLAS_ANNOTATED',bool(final))

if __name__=='__main__':main()

"""Incremental v34 provenance: auxiliary NLL is not an official TOTAL."""
from pathlib import Path
from datetime import datetime
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STUDY=ROOT/'results/deep_learning/cross_target_v34_20260929'

def main():
    path=HERE/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    final_path=STUDY/'assembled_report.json';final=json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else None
    parents=[]
    for g in range(4):
        key=f'pure_dl_cross_target_v34_aux{g}';report_path=ROOT/'outputs'/key/'auxiliary_report.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text(encoding='utf-8'))
        desc='；'.join(f"{x['target']} NLL={x['stop_NLL']:.6f}@{x['best_epoch']}" for x in report['models'])
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title=f'v34 辅助交叉拟合组{g}',stage='training',status='recorded',highlight=False,
            summary='40井拟合/8井辅助NLL早停/16井OOF，三个独立9881参数MLP；原第五折完全隔离。',
            conclusion=desc+'。这些是辅助NLL，不是第五折或完整OOF官方分数；预测供冻结SW小头使用。',
            primaryUnavailableWhy='独立辅助模型NLL，不能混排官方TOTAL',
            evidence=[str(report_path.relative_to(ROOT)),f'results/deep_learning/cross_target_v34_20260929/aux_config_g{g}.json'])
        parents.append(key)
    key='pure_dl_cross_target_v34_SW_fold5';out=ROOT/'outputs'/key
    status_path=out/'runtime_status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        complete=status.get('status')=='complete'
        score=max([73.6630827970656]+[v['best'] for v in status.get('modes',{}).values()])
        if final:score=final['selected_scores']['SW']
        source='outputs/'+key+('/fold_5_report.json' if complete else '/runtime_status.json')
        desc='self/cross/shuffled三个同容量权重头；父模型冻结，辅助源为井外OOF。'
        if final:
            desc+='；'.join(f"{r['mode']}更新后SW={r['post_update_scores']['SW']:.9f}" for r in final['candidates'])
            desc+=f"。实际TOTAL={final['selected_scores']['TOTAL']:.12f}，跨目标机制通过={final['cross_target_mechanism_positive']}。"
        else:desc+='训练中，当前快照包含第0轮；尚未实际组合。'
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title='v34 跨目标预测协同',stage='architecture',status='recorded' if complete else 'running',
            highlight=bool(final and final['adopted']),summary='保留强SW；增加12维同目标/跨目标/打乱对应描述，三个11152参数头。',
            primary=dict(value=score,kind='diagnostic',scope='原第五折 · SW单通道',source=source),conclusion=desc,
            evidence=[source,'results/deep_learning/cross_target_v34_20260929/METHOD.md','results/deep_learning/cross_target_v34_20260929/feature_manifest.json'])
        for parent in parents+['pure_dl_split_route_v29_SW_fold5']:
            edge=dict(source=parent,target=key,type='evolution',note='辅助预测输入依赖；SW父模型保持，非标签投票',evidence=source)
            if not any(e.get('source')==parent and e.get('target')==key for e in cur.setdefault('edges',[])):cur['edges'].append(edge)
        dead='dead:cross_target_v34_small_aux'
        cur['deadends']=[d for d in cur.get('deadends',[]) if d.get('id')!=dead]
        if final and not final['cross_target_mechanism_positive']:
            cur['deadends'].append(dict(id=dead,title='v34 微型辅助预测未形成协同增量',stage='architecture',parents=[key],summary=desc,
                reason='仅约束本次小型辅助来源、6维描述和SW权重修正，不否定强辅助模型或其他跨目标机制；不追加无界调参。'))
    tmp=path.with_suffix('.v34.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    marker='## v34跨目标证据图谱'+('完成' if final else '进行中')
    p=HERE/'PROJECTS.md'
    if marker not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n'+datetime.now().astimezone().isoformat()+'；辅助NLL与官方SW分数分开，原curate.py及历史训练源保持。\n')
    print('V34_ATLAS',json.dumps(dict(final=bool(final),auxiliary_groups=len(parents))),flush=True)

if __name__=='__main__':main()

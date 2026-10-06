"""Read-only experiment observation plus optional documented graph metadata."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_pyramid_v19_20260928'
MODES=('regional','global')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def trim(d):
    return {k:v for k,v in d.items() if k!='per_well'}


def record_graph(result):
    path=ROOT/'docs/deep_learning/atlas/curation.json';graph=load(path)
    for row in result['rows']:
        done=row['final'];score=done['target_score'] if done else row['status'].get('best_target_score')
        source='outputs/'+row['id']+('/fold_5_report.json' if done else '/runtime_status.json')
        primary=dict(value=score,scope='原第五折 · POR单通道'+('' if done else ' · 未完成'),kind='diagnostic',source=source) if score is not None else None
        conclusion=(f"完成POR={score:.9f}，最佳{done['best_epoch']}轮；实际合并TOTAL见研究assembled_report" if done else '运行快照，尚无完成成绩')
        graph.setdefault('nodes',{})[row['id']]=dict(title='v19多尺度POR '+('区域记忆对照' if row['mode']=='regional' else '全窗口记忆'),
            stage='architecture',summary='96维256/128/64金字塔+双跳接+门控潜变量辅助，两组同参数同初始化，只改注意力连通范围。',
            status='recorded' if done else row['status']['status'],primary=primary,conclusion=conclusion,
            evidence=[source,'outputs/'+row['id']+'/README.md','results/deep_learning/por_pyramid_v19_20260928/METHOD.md'])
    edges=[dict(source='pure_dl_por_architecture_v18_tcn104_fold5',target=result['rows'][0]['id'],type='evolution',note='完整多尺度骨干，容量也增长；非纯单因素比较'),
        dict(source=result['rows'][0]['id'],target=result['rows'][1]['id'],type='control',note='同参数同初始状态，仅记忆连通范围为区域或全窗口')]
    for edge in edges:
        if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in graph.setdefault('edges',[])):graph['edges'].append(edge)
    path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0)
    args=ap.parse_args()
    if args.wait:
        jobs=[]
        for m in MODES:
            s=load(ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{m}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_por_pyramid_v19.py' in a for a in p.cmdline()):jobs.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(jobs,timeout=min(max(args.wait,0),40))
    rows=[];configs=[]
    for m in MODES:
        out=ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{m}_fold5'
        status=load(out/'runtime_status.json') or dict(status='not_started')
        cfg=load(out/'config.json');configs.append(cfg)
        final=load(out/'fold_5_report.json');history=load(out/'validation_history.json') or []
        if final:
            final=dict(final);final['validation']=trim(final['validation']);final['fit']=trim(final['fit'])
        latest=dict(history[-1]) if history else None
        if latest:latest['validation']=trim(latest['validation'])
        rows.append(dict(mode=m,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
            parameter_count=cfg['trainable_parameters'] if cfg else None,initial_hash=cfg.get('initial_state_sha256') if cfg else None,
            final=final,latest=latest,gradient_scope=load(out/'gradient_scope.json'),failure=load(out/'failure.json')))
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(r['final'] is not None for r in rows),
        assembly_complete=(STUDY/'assembled_report.json').exists(),complete_five_folds=False,remaining_folds_started=False)
    if all(configs):
        diff=[k for k in sorted(set(configs[0])|set(configs[1])) if configs[0].get(k)!=configs[1].get(k)]
        result.update(config_differences=diff,identical_parameters=rows[0]['parameter_count']==rows[1]['parameter_count'],
            identical_initial_state=rows[0]['initial_hash']==rows[1]['initial_hash'])
        assert result['identical_parameters'] and result['identical_initial_state']
        assert set(diff)<= {'mode','memory_connectivity'}
    if result['all_complete']:
        result['global_minus_regional_POR']=rows[1]['final']['target_score']-rows[0]['final']['target_score']
        result['global_minus_regional_TOTAL']=.3*result['global_minus_regional_POR']
        result['initial_predictions_identical']=bool(np.array_equal(np.load(ROOT/'outputs'/rows[0]['id']/'initial_pred.npy'),np.load(ROOT/'outputs'/rows[1]['id']/'initial_pred.npy')))
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.record:record_graph(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

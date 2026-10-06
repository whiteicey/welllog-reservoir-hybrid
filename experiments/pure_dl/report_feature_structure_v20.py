"""Observe bounded v20 runs and maintain current graph metadata; never launch."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/feature_structure_v20_20260928'
MODES=('amplitude_early','shape_early','shape_late')

def load(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None

def trim(x):return {k:v for k,v in x.items() if k!='per_well'}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0);a=ap.parse_args()
    if a.wait:
        jobs=[]
        for m in MODES:
            s=load(ROOT/'outputs'/f'pure_dl_feature_structure_v20_{m}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_feature_structure_v20.py' in x for x in p.cmdline()):jobs.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(jobs,timeout=min(max(a.wait,0),40))
    rows=[];configs=[]
    for m in MODES:
        out=ROOT/'outputs'/f'pure_dl_feature_structure_v20_{m}_fold5'
        cfg=load(out/'config.json');configs.append(cfg)
        s=load(out/'runtime_status.json') or dict(status='not_started')
        final=load(out/'fold_5_report.json');hist=load(out/'validation_history.json') or []
        if final:
            final=dict(final);final['validation']=trim(final['validation']);final['fit']=trim(final['fit'])
        latest=dict(hist[-1]) if hist else None
        if latest:latest['validation']=trim(latest['validation'])
        row=dict(mode=m,id=out.name,status=s,pid_alive=psutil.pid_exists(s.get('pid',-1)),
            parameter_count=cfg['trainable_parameters'] if cfg else None,initial_hash=cfg.get('initial_state_sha256') if cfg else None,
            final=final,latest=latest,gradient_scope=load(out/'gradient_scope.json'),failure=load(out/'failure.json'))
        rows.append(row)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(r['final'] is not None for r in rows),
        assembly_complete=(STUDY/'assembled_report.json').exists(),complete_five_folds=False,remaining_folds_started=False)
    ready=[r for r in rows if r['initial_hash']]
    if len(ready)>1:
        result['same_initial_state']=len({r['initial_hash'] for r in ready})==1
        result['same_parameters']=len({r['parameter_count'] for r in ready})==1
        assert result['same_initial_state'] and result['same_parameters']
    if result['all_complete']:
        result['feature_delta_POR']=rows[1]['final']['target_score']-rows[0]['final']['target_score']
        result['fusion_delta_POR']=rows[2]['final']['target_score']-rows[1]['final']['target_score']
        differences={}
        for i,j,name in ((0,1,'feature'),(1,2,'fusion')):
            differences[name]=[k for k in sorted(configs[i]) if configs[i].get(k)!=configs[j].get(k)]
            assert set(differences[name])<=({'mode','side_view'} if name=='feature' else {'mode','fusion'})
        result['config_differences']=differences
        initial=[np.load(ROOT/'outputs'/r['id']/'initial_pred.npy') for r in rows]
        result['same_initial_predictions']=all(np.array_equal(initial[0],x) for x in initial[1:])
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.record:
        path=ROOT/'docs/deep_learning/atlas/curation.json';graph=load(path)
        titles=('幅值早融合','形态早融合','形态分支后融合')
        for r,title in zip(rows,titles):
            if r['status']['status']=='not_started':continue
            done=r['final'];score=done['target_score'] if done else r['status'].get('best_target_score')
            source='outputs/'+r['id']+('/fold_5_report.json' if done else '/runtime_status.json')
            primary=dict(value=score,scope='原第五折 · POR单通道'+('' if done else ' · 运行中'),kind='diagnostic',source=source) if score is not None else None
            graph.setdefault('nodes',{})[r['id']]=dict(title='v20 '+title,stage='architecture',
                status='recorded' if done else r['status']['status'],primary=primary,
                summary='原206维+63维输入侧视图，双80维编码器；同参数初始状态，分开比较特征与融合位置。',
                conclusion=f"完成POR={score:.9f}，实际三通道总分见研究assembled_report" if done else '运行中，尚无最终收益',
                evidence=[source,'outputs/'+r['id']+'/README.md','results/deep_learning/feature_structure_v20_20260928/feature_manifest.json'])
        edges=[('pure_dl_por_architecture_v18_tcn104_fold5',rows[0]['id'],'evolution','增加输入侧视图并采用双分支骨干，参数不同'),
            (rows[0]['id'],rows[1]['id'],'control','同结构同参数，只改变63维侧视图的幅值/标准化形态内容'),
            (rows[1]['id'],rows[2]['id'],'control','相同输入和全部参数，仅改变两视图混合位置')]
        for src,dst,kind,note in edges:
            if not any(e['source']==src and e['target']==dst for e in graph.setdefault('edges',[])):graph['edges'].append(dict(source=src,target=dst,type=kind,note=note))
        path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

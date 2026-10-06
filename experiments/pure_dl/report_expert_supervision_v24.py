"""Read v24 state, maintain result scope, never launch or stop a trainer."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/expert_supervision_v24_20260928'
MODES=('mixture','independent')


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None


def save(p,value):
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0)
    args=ap.parse_args()
    if args.wait:
        processes=[]
        for m in MODES:
            s=load(ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{m}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_expert_supervision_v24.py' in a for a in p.cmdline()):processes.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(processes,timeout=min(max(args.wait,0),40))
    rows=[];cfgs=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{mode}_fold5'
        cfg=load(out/'config.json');cfgs.append(cfg)
        status=load(out/'runtime_status.json') or {'status':'not_started'}
        h=load(out/'validation_history.json') or []
        final=load(out/'fold_5_report.json')
        row=dict(mode=mode,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
            parameter_count=cfg['trainable_parameters'] if cfg else None,initial=load(out/'initial_verification.json'),
            gradient_scope=load(out/'gradient_scope.json'),latest=h[-1] if h else None,
            final=final,failure=load(out/'failure.json'))
        if row['failure']:
            row['stderr_tail']=(out/'train.err').read_text(encoding='utf-8',errors='replace')[-2500:]
        rows.append(row)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['final'] is not None for r in rows),assembly_complete=(STUDY/'assembled_report.json').exists(),
        complete_five_folds=False,remaining_folds_started=False)
    if all(cfgs):
        differences=[k for k in sorted(set(cfgs[0])|set(cfgs[1])) if cfgs[0].get(k)!=cfgs[1].get(k)]
        assert set(differences)=={'mode','expert_objective'},differences
        result['config_differences']=differences
        result['same_parameters_and_initial_state']=True
    initial=[ROOT/'outputs'/r['id']/'initial_pred.npy' for r in rows]
    if all(p.exists() for p in initial):
        result['all144665_initial_predictions_equal']=bool(np.array_equal(np.load(initial[0]),np.load(initial[1])))
        assert result['all144665_initial_predictions_equal']
    if result['all_complete']:
        result['independent_minus_mixture_POR']=rows[1]['final']['target_score']-rows[0]['final']['target_score']
        result['independent_minus_mixture_TOTAL']=.3*result['independent_minus_mixture_POR']
    save(STUDY/'progress.json',result)
    if args.record:
        path=ROOT/'docs/deep_learning/atlas/curation.json';g=load(path)
        for r in rows:
            done=r['final'];score=done['target_score'] if done else r['status'].get('best_target_score')
            source='outputs/'+r['id']+('/fold_5_report.json' if done else '/runtime_status.json')
            primary=dict(value=score,scope='原第五折 · POR单通道'+('' if done else ' · 运行中'),kind='diagnostic',source=source) if score is not None else None
            g.setdefault('nodes',{})[r['id']]=dict(title='v24 '+('混合责任监督' if r['mode']=='mixture' else '专家独立监督'),
                stage='loss',status='recorded' if done else r['status']['status'],primary=primary,
                summary='1290490参数相同双专家、相同初始化，两组都独立训练路由；只改专家损失。',
                conclusion=f"完成POR={score:.9f}，实际总分和匹配差见expert_supervision_v24_20260928/assembled_report.json" if done else '尚未完成，不声明训练收益',
                evidence=[source,'outputs/'+r['id']+'/README.md'])
        edges=[dict(source='pure_dl_scratch_scaling_v23_base1_fold5',target=rows[0]['id'],type='evolution',note='同网络与随机起点，改成专家/路由独立优化和分组裁剪；新匹配对照'),
            dict(source=rows[0]['id'],target=rows[1]['id'],type='control',note='同参数和训练配置，仅专家损失由混合改为各自完整似然的均值')]
        for e in edges:
            if not any(v['source']==e['source'] and v['target']==e['target'] for v in g.setdefault('edges',[])):g['edges'].append(e)
        save(path,g)
    compact={k:v for k,v in result.items() if k!='rows'}
    compact['rows']=[]
    for r in rows:
        v={k:r[k] for k in ('mode','status','pid_alive','parameter_count','initial','gradient_scope','latest','failure')}
        if r['final']:v['final']={k:x for k,x in r['final'].items() if k not in ('best_after_update','initial','gradient_scope')}
        if 'stderr_tail' in r:v['stderr_tail']=r['stderr_tail']
        compact['rows'].append(v)
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

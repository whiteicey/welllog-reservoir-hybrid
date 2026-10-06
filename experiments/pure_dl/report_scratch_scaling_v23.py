"""Bounded v23 observation and evidence-based graph records; never starts jobs."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/scratch_scaling_v23_20260928'
MODES=('base1','ffn2')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0)
    args=ap.parse_args()
    if args.wait:
        jobs=[]
        for mode in MODES:
            s=load(ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_scratch_scaling_v23.py' in a for a in p.cmdline()):jobs.append(p)
                except(psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(jobs,timeout=min(max(args.wait,0),40))
    rows=[];configs=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5'
        cfg=load(out/'config.json');configs.append(cfg)
        status=load(out/'runtime_status.json') or dict(status='not_started')
        history=load(out/'validation_history.json') or []
        final=load(out/'fold_5_report.json')
        row=dict(mode=mode,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
            initial=load(out/'initial_verification.json'),gradient_scope=load(out/'gradient_scope.json'),
            parameter_count=cfg['trainable_parameters'] if cfg else None,
            common_initial_state=cfg['scaling']['common_initial_state_sha256'] if cfg else None,
            latest=history[-1] if history else None,final=final,failure=load(out/'failure.json'))
        if row['failure']:
            row['stderr_tail']=(out/'train.err').read_text(encoding='utf-8',errors='replace')[-3000:]
        rows.append(row)
    summary=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(r['final'] for r in rows),
        assembly_complete=(STUDY/'assembled_report.json').exists(),complete_five_folds=False,remaining_folds_started=False)
    if all(configs):
        keys=('epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','learning_rate','eta_min','weight_decay','clip_norm','quality','model_metadata','cache','objective','decoder')
        assert all(configs[0][k]==configs[1][k] for k in keys)
        assert rows[0]['common_initial_state']==rows[1]['common_initial_state']
        summary['common_random_state_and_training_protocol_match']=True
        summary['parameter_ratio']=rows[1]['parameter_count']/rows[0]['parameter_count']
    paths=[ROOT/'outputs'/r['id']/'initial_pred.npy' for r in rows]
    if all(p.exists() for p in paths):
        summary['all144665_initial_predictions_equal']=bool(np.array_equal(np.load(paths[0]),np.load(paths[1])))
        assert summary['all144665_initial_predictions_equal']
    if summary['all_complete']:
        summary['ffn2_minus_base1_POR']=rows[1]['final']['target_score']-rows[0]['final']['target_score']
        summary['ffn2_minus_base1_TOTAL']=summary['ffn2_minus_base1_POR']*.3
    save(STUDY/'progress.json',summary)
    if args.record:
        path=ROOT/'docs/deep_learning/atlas/curation.json';graph=load(path)
        for r in rows:
            report=r['final'];s=report['target_score'] if report else r['status'].get('best_target_score')
            source='outputs/'+r['id']+('/fold_5_report.json' if report else '/runtime_status.json')
            primary=dict(value=s,scope='原第五折 · POR单通道'+('' if report else ' · 运行中'),kind='diagnostic',source=source) if s is not None else None
            graph.setdefault('nodes',{})[r['id']]=dict(title='v23从头双专家 '+r['mode'],stage='architecture',
                status='recorded' if report else r['status']['status'],primary=primary,
                summary='相同双视图/多尺度骨干，从随机权重学习POR，129万/198万容量；固定Q50及单任务最终混合损失。',
                conclusion=f"完成POR={s:.9f}，实际TOTAL见scratch_scaling_v23_20260928/assembled_report.json" if report else '从头容量对照运行中，尚未确认收益',
                evidence=[source,'outputs/'+r['id']+'/README.md'])
        edges=[dict(source='pure_dl_frozen_growth_v22_ffn2_fold5',target=rows[0]['id'],type='evolution',note='改为随机网络/共同从头训练学习率；不是单因素热启动比较'),
            dict(source=rows[0]['id'],target=rows[1]['id'],type='control',note='相同随机原参数、数据和训练协议，仅扩大内部FFN与位置头容量')]
        for edge in edges:
            if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in graph.setdefault('edges',[])):graph['edges'].append(edge)
        save(path,graph)
    concise={k:v for k,v in summary.items() if k!='rows'}
    concise['rows']=[{k:v for k,v in r.items() if k!='final'}|({'final':{k:v for k,v in r['final'].items() if k not in ('scaling','best_after_update')}} if r['final'] else {}) for r in rows]
    print(json.dumps(concise,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

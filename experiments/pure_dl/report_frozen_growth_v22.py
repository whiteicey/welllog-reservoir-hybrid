"""Observe v22 without starting/stopping any training; record real result scopes."""
from pathlib import Path
from datetime import datetime
import argparse, json, sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/frozen_growth_v22_20260928'
MODES=('ffn2','ffn4')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser(); ap.add_argument('--record',action='store_true'); ap.add_argument('--wait',type=int,default=0)
    args=ap.parse_args()
    if args.wait:
        processes=[]
        for m in MODES:
            status=read_json(ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{m}_fold5/runtime_status.json')
            if status and status['status']=='running':
                try:
                    process=psutil.Process(status['pid'])
                    if any('train_frozen_growth_v22.py' in s for s in process.cmdline()): processes.append(process)
                except(psutil.NoSuchProcess,psutil.AccessDenied): pass
        psutil.wait_procs(processes,timeout=min(max(args.wait,0),40))
    rows=[]
    for m in MODES:
        out=ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{m}_fold5'
        status=read_json(out/'runtime_status.json') or dict(status='not_started')
        cfg=read_json(out/'config.json')
        history=read_json(out/'validation_history.json') or []
        final=read_json(out/'fold_5_report.json')
        previous=read_json(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{m}_fold5/fold_5_report.json')
        previous_cfg=read_json(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{m}_fold5/config.json')
        row=dict(mode=m,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
            total_parameters=cfg['total_parameters'] if cfg else None,trainable_parameters=cfg['trainable_parameters'] if cfg else None,
            initial=read_json(out/'initial_verification.json'),gradient_scope=read_json(out/'gradient_scope.json'),
            latest=history[-1] if history else None,
            best_after_update=max(history,key=lambda r:r['target_score']) if history else None,
            final=final,failure=read_json(out/'failure.json'))
        if cfg and previous_cfg:
            keys=('epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','backbone_lr','readout_lr','eta_min','weight_decay','clip_norm','quality')
            different=[k for k in keys if cfg[k]!=previous_cfg[k]]
            assert not different, different
            assert cfg['total_parameters']==previous_cfg['trainable_parameters']
            row['same_v21_architecture_budget_quality_and_lr']=True
        if previous:
            row['v21_best_after_update']={k:previous['best_after_update'][k] for k in ('epoch','target_score','reference_replacement_total')}
            if row['best_after_update']:
                row['delta_v21_best_updated_TOTAL']=row['best_after_update']['reference_replacement_total']-previous['best_after_update']['reference_replacement_total']
        if row['failure']:
            err=out/'train.err'
            row['stderr_tail']=err.read_text(encoding='utf-8',errors='replace')[-3500:] if err.exists() else None
        rows.append(row)
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
        all_complete=all(r['final'] is not None for r in rows),assembly_complete=(STUDY/'assembled_report.json').exists(),
        complete_five_folds=False,remaining_folds_started=False)
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.record:
        path=ROOT/'docs/deep_learning/atlas/curation.json'
        graph=read_json(path)
        for row in rows:
            done=row['final'];score=done['target_score'] if done else row['status'].get('best_target_score')
            source='outputs/'+row['id']+('/fold_5_report.json' if done else '/runtime_status.json')
            primary=dict(value=score,scope='原第五折 · POR单通道'+('' if done else ' · 运行快照'),kind='diagnostic',source=source) if score is not None else None
            conclusion=(f"完成POR={score:.9f}，最佳第{done['best_epoch']}轮；实际组合总分见研究assembled_report。" if done else '运行中，不代表完成收益')
            graph.setdefault('nodes',{})[row['id']]=dict(title='v22仅新增容量 '+row['mode'],stage='architecture',
                status='recorded' if done else row['status']['status'],primary=primary,
                summary='沿用v21内部扩容，原1290490参数固定，只训练额外FFN和位置头；原数据/Q/LR不变。',
                conclusion=conclusion,evidence=[source,'outputs/'+row['id']+'/README.md'])
            edge=dict(source='pure_dl_champion_scaling_v21_'+row['mode']+'_fold5',target=row['id'],type='control',note='相同扩容结构与初始化，仅冻结所有原有参数')
            if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in graph.setdefault('edges',[])): graph['edges'].append(edge)
        path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

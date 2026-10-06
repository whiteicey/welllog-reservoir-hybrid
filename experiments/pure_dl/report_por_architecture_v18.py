"""Read and optionally record v18 results. No training is started or stopped."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_architecture_v18_20260928'
MODES=('tcn104','latent64')

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None

def trim(stats):
    return {k:v for k,v in stats.items() if k!='per_well'}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0)
    args=ap.parse_args()
    if args.wait:
        procs=[]
        for m in MODES:
            s=load(ROOT/'outputs'/f'pure_dl_por_architecture_v18_{m}_fold5/runtime_status.json')
            if s and s['status']=='running':
                try:
                    p=psutil.Process(s['pid'])
                    if any('train_por_architecture_v18.py' in a for a in p.cmdline()):procs.append(p)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.wait_procs(procs,timeout=min(max(args.wait,0),40))
    rows=[]
    for m in MODES:
        out=ROOT/'outputs'/f'pure_dl_por_architecture_v18_{m}_fold5'
        status=load(out/'runtime_status.json') or {'status':'not_started'}
        config=load(out/'config.json');final=load(out/'fold_5_report.json');history=load(out/'validation_history.json') or []
        if final:
            final=dict(final);final['validation']=trim(final['validation']);final['fit']=trim(final['fit'])
        latest=dict(history[-1]) if history else None
        if latest:latest['validation']=trim(latest['validation'])
        rows.append(dict(mode=m,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
            parameter_count=config['trainable_parameters'] if config else None,
            initial_bias_hash=config['initial_output_bias_sha256'] if config else None,
            initial=trim(load(out/'initial.json')['validation']) if (out/'initial.json').exists() else None,
            gradient_scope=load(out/'gradient_scope.json'),final=final,latest=latest,failure=load(out/'failure.json')))
    result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,all_complete=all(x['final'] for x in rows),
        assembly_complete=(STUDY/'assembled_report.json').exists(),remaining_folds_started=False,complete_five_folds=False)
    if all(x['parameter_count'] for x in rows):
        result['parameter_difference']=rows[1]['parameter_count']-rows[0]['parameter_count']
        result['same_initial_output_bias']=rows[0]['initial_bias_hash']==rows[1]['initial_bias_hash']
    if result['all_complete']:
        result['latent_minus_tcn_POR']=rows[1]['final']['target_score']-rows[0]['final']['target_score']
        result['latent_minus_tcn_TOTAL']=.3*result['latent_minus_tcn_POR']
        result['initial_predictions_exactly_equal']=np.array_equal(np.load(ROOT/'outputs'/rows[0]['id']/'initial_pred.npy'),np.load(ROOT/'outputs'/rows[1]['id']/'initial_pred.npy'))
    (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.record:
        path=ROOT/'docs/deep_learning/atlas/curation.json';graph=load(path)
        for row in rows:
            final=row['final'];score=final['target_score'] if final else row['status'].get('best_target_score')
            source='outputs/'+row['id']+('/fold_5_report.json' if final else '/runtime_status.json')
            primary=dict(value=score,scope='原第五折 · POR单通道'+('' if final else ' · 运行中'),kind='diagnostic',source=source) if score is not None else None
            text=(f"完成POR={score:.9f}，最佳{final['best_epoch']}轮；三通道实测以assembled_report为准" if final else '本批运行快照，尚无最终成绩')
            graph.setdefault('nodes',{})[row['id']]=dict(title='v18 POR '+('宽TCN104' if row['mode']=='tcn104' else '潜变量记忆64'),
                stage='architecture',status='recorded' if final else row['status']['status'],primary=primary,
                summary='相近16万参数、同Student-t3输出，从随机网络学习POR；固定PERM/SW。',conclusion=text,
                evidence=[source,'outputs/'+row['id']+'/README.md'])
            edge=dict(source='pure_dl_latent_v16_latent_fold5',target=row['id'],type='evolution',note='全局记忆骨干扩展到POR，新目标重新训练')
            if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in graph.setdefault('edges',[])):graph['edges'].append(edge)
        graph['edges']=[e for e in graph['edges'] if not (e['target']==rows[0]['id'] and e['source']=='pure_dl_latent_v16_latent_fold5')]
        edge=dict(source='pure_dl_por_architecture_v18_tcn104_fold5',target='pure_dl_por_architecture_v18_latent64_fold5',type='control',note='近参数量、共同POR头，局部与全局上下文方案比较')
        if not any(e['source']==edge['source'] and e['target']==edge['target'] for e in graph['edges']):graph['edges'].append(edge)
        path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

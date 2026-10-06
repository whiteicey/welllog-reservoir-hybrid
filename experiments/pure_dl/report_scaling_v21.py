"""Observe scaling experiments; record scores with and without epoch zero."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import psutil
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_scaling_v21_20260928'
MODES=('base1','ffn2','ffn4')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else None

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--wait',type=int,default=0);args=ap.parse_args()
 if args.wait:
  jobs=[]
  for m in MODES:
   s=read(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{m}_fold5/runtime_status.json')
   if s and s['status']=='running':
    try:
     p=psutil.Process(s['pid'])
     if any('train_scaling_v21.py' in a for a in p.cmdline()):jobs.append(p)
    except(psutil.NoSuchProcess,psutil.AccessDenied):pass
  psutil.wait_procs(jobs,timeout=min(max(args.wait,0),40))
 rows=[];configs=[]
 for mode in MODES:
  out=ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5'
  status=read(out/'runtime_status.json') or {'status':'not_started'}
  config=read(out/'config.json');configs.append(config)
  history=read(out/'validation_history.json') or []
  final=read(out/'fold_5_report.json');initial=read(out/'initial_verification.json')
  if final:
   final=dict(final);final['scaling']={k:v for k,v in final['scaling'].items() if k not in ('blocks','dropout_modules')}
  best_post=max(history,key=lambda r:r['target_score']) if history else None
  row=dict(mode=mode,id=out.name,status=status,pid_alive=psutil.pid_exists(status.get('pid',-1)),
   parameters=config['trainable_parameters'] if config else None,
   added_parameters=config['scaling']['extra_parameters'] if config else None,
   location_hidden_width=config['scaling']['location_hidden_width'] if config else None,
   initial=initial,gradient_scope=read(out/'gradient_scope.json'),best_after_update=best_post,
   last=history[-1] if history else None,final=final,failure=read(out/'failure.json'))
  if config:row['iteration_seconds_median']=float(__import__('statistics').median([v['seconds'] for v in history])) if history else None
  rows.append(row)
 result=dict(observed_at=datetime.now().astimezone().isoformat(),rows=rows,
  all_complete=all(r['final'] is not None for r in rows),assembly_complete=(STUDY/'assembled_report.json').exists(),
  complete_five_folds=False,remaining_folds_started=False)
 if all(c is not None for c in configs):
  diff={}
  for cfg in configs[1:]:
   fields=[k for k in configs[0] if configs[0].get(k)!=cfg.get(k)]
   assert set(fields)<={'mode','scaling','trainable_parameters'},fields
   diff[cfg['mode']]=fields
  result['config_differences']=diff
 result['same_initial_function_verified']=all(r['initial'] is not None and r['initial']['different_target_elements_vs_shared_reference']==0 for r in rows)
 if result['all_complete']:
  control=rows[0]['final']['reference_replacement_total']
  control_post=rows[0]['best_after_update']['reference_replacement_total']
  result['comparisons']=[dict(mode=r['mode'],best_including_epoch0_TOTAL=r['final']['reference_replacement_total'],
   delta_control_selected=r['final']['reference_replacement_total']-control,
   best_after_update_TOTAL=r['best_after_update']['reference_replacement_total'],
   delta_control_after_update=r['best_after_update']['reference_replacement_total']-control_post) for r in rows]
 (STUDY/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 if args.record:
  path=ROOT/'docs/deep_learning/atlas/curation.json';graph=read(path)
  for row in rows:
   if row['status']['status']=='not_started':continue
   done=row['final'];score=done['reference_replacement_total'] if done else row['status'].get('best_target_score')
   if not done and score is not None:score=80.27413533779057+.3*(score-77.44891570750882)
   source='outputs/'+row['id']+('/fold_5_report.json' if done else '/runtime_status.json')
   primary=dict(value=score,scope='原第五折 · 固定PERM/SW替换TOTAL'+('' if done else ' · 未完成'),kind='diagnostic',source=source) if score is not None else None
   post=row['best_after_update']
   graph.setdefault('nodes',{})[row['id']]=dict(title='v21冠军扩容 '+row['mode'],stage='architecture',
    status='recorded' if done else row['status']['status'],primary=primary,
    summary='POR双专家内部非线性宽度1/2/4倍，零输出增长保持冠军起点，全部POR私有参数训练。',
    conclusion=('完成；保留最佳epoch='+str(done['best_epoch'])+'，更新后最好TOTAL='+str(post['reference_replacement_total'])) if done else '运行快照，收益未确认',
    evidence=[source,'outputs/'+row['id']+'/README.md','results/deep_learning/champion_scaling_v21_20260928/gain_attribution.json'])
  for src,dst,kind in [('pure_dl_independent_v8_POR_dual80_fold5',rows[0]['id'],'evolution'),(rows[0]['id'],rows[1]['id'],'control'),(rows[1]['id'],rows[2]['id'],'control')]:
   if not any(e['source']==src and e['target']==dst for e in graph.setdefault('edges',[])):
    graph['edges'].append(dict(source=src,target=dst,type=kind,note='同冠军起点，扩容内部非线性；同数据/学习率/预算'))
  path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""CPU terminal evidence, never launches training or auto-promotes."""
import sys,json,math
from pathlib import Path
import torch,numpy as np
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,2]
source=torch.load(ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt',map_location='cpu',weights_only=False)['state_dict']
reports={};strata={};norms={};ends={}
for arm in ('control','adapter'):
 out=ROOT/f'outputs/pure_dl_sw_native_joint_adapter_v319_fold5/{arm}'
 rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
 assert (STUDY/f'{arm}_assembled.json').exists()
 reports[arm]=rep;ends[arm]=(out/'report.json').stat().st_mtime
 pred=y.copy();pred[:,2]=np.load(out/'channel_pred.npy')
 strata[arm]={n:float(base.metric(y[m],pred[m])['SW']) for n,m in [('default',dm),('continuous',~dm)]}
 state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];groups={}
 for n,v in state.items():
  key=n.removeprefix('base.').replace('body.global_blocks.3.ff.old.','body.global_blocks.3.ff.')
  if key not in source:continue
  z=v.float()-source[key].float()
  if key.startswith('output.'):
   groups['state']=groups.get('state',0)+float(z[:1].square().sum());groups['density']=groups.get('density',0)+float(z[1:].square().sum())
  else:
   g='condition' if key.startswith('condition.') else 'body_norm';groups[g]=groups.get(g,0)+float(z.square().sum())
 norms[arm]={k:math.sqrt(v) for k,v in groups.items()}
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Concurrent pair with21.527s stagger; individual includes preparation/initial/train/finalrestore; pair includes creation-to-last-terminal, excludes CPUfinalization. No isolatedspeed comparison.')
result=dict(reports=reports,strata=strata,best_source_parameter_delta_L2=norms,adapter_minus_control=reports['adapter']['target_score']-reports['control']['target_score'],adapter_minus_control_CONT=strata['adapter']['continuous']-strata['control']['continuous'],gain_vs297={a:r['target_score']-74.16003022301975 for a,r in reports.items()},CONT_vs297={a:v['continuous']-17.502752767347182 for a,v in strata.items()},CONT_vs252={a:v['continuous']-17.88486813991017 for a,v in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,adopted=False,capacity_matched=False)
for name,value in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]: (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
print(json.dumps(result))

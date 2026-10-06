"""CPU terminal selective-memory paired evidence, no launch or promotion."""
from pathlib import Path
import sys,json,numpy as np,torch
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
from perm_selective_memory_v325 import SelectiveMemory
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,1]
reports={};strata={};norms={};ends={}
for arm in ('selective','window'):
 out=ROOT/f'outputs/pure_dl_perm_selective_memory_v325_fold5/{arm}';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
 assert (STUDY/f'{arm}_assembled.json').exists();reports[arm]=rep;ends[arm]=(out/'report.json').stat().st_mtime
 pred=y.copy();pred[:,1]=np.load(out/'channel_pred.npy');strata[arm]={n:float(base.metric(y[m],pred[m])['PERM']) for n,m in [('default',dm),('continuous',~dm)]}
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20261325);initial=SelectiveMemory(arm)
 state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms[arm]={n:float((state['depth_memory.'+n].float()-v.float()).norm()) for n,v in initial.state_dict().items()}
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Actual stagger/concurrency retained. Includes data/initial/train/bestrestore, excludes implementation/CPUfinalization; not isolatedspeed ranking.')
result=dict(reports=reports,strata=strata,best_memory_delta_norms=norms,selective_minus_window=reports['selective']['target_score']-reports['window']['target_score'],selective_minus_window_CONT=strata['selective']['continuous']-strata['window']['continuous'],gain_vs312={a:r['target_score']-89.51580440487884 for a,r in reports.items()},CONT_vs312={a:v['continuous']-74.00288840401272 for a,v in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,adopted=False)
for n,v in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]: (STUDY/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(result))

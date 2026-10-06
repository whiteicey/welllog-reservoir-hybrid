"""CPU terminal matched latent-morphology evidence, no automatic adoption."""
from pathlib import Path
import sys,json,numpy as np,torch
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
from sw_latent_morphology_v326 import Morphology
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,2]
reports={};strata={};norms={};ends={}
for arm in ('depth','channel'):
 out=ROOT/f'outputs/pure_dl_sw_latent_morphology_v326_fold5/{arm}';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
 assert (STUDY/f'{arm}_assembled.json').exists();reports[arm]=rep;ends[arm]=(out/'report.json').stat().st_mtime
 pred=y.copy();pred[:,2]=np.load(out/'channel_pred.npy');strata[arm]={n:float(base.metric(y[m],pred[m])['SW']) for n,m in [('default',dm),('continuous',~dm)]}
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20261326);initial=Morphology(arm)
 state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms[arm]={n:float((state['morphology.'+n].float()-v.float()).norm()) for n,v in initial.state_dict().items()}
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Actual stagger/concurrency retained; includes preparation/initial/train/bestrestoration, excludes implementation/CPUfinalization. Not isolatedspeed ranking.')
result=dict(reports=reports,strata=strata,best_morphology_delta_norms=norms,depth_minus_channel=reports['depth']['target_score']-reports['channel']['target_score'],depth_minus_channel_CONT=strata['depth']['continuous']-strata['channel']['continuous'],gain_vs321own={a:r['target_score']-74.16603018049169 for a,r in reports.items()},CONT_vs321own={a:v['continuous']-17.911355427689944 for a,v in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,adopted=False)
for n,v in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]: (STUDY/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(result))

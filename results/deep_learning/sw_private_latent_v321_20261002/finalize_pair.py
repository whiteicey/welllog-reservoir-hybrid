"""CPU terminal paired evidence, no launch or automatic adoption."""
from pathlib import Path
import sys,json
import torch,numpy as np
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
from sw_private_latent_v321r1 import Network
from torch import nn
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(20261321);initial=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,48),nn.GELU(),nn.Linear(48,48));nn.init.zeros_(initial[-1].weight);nn.init.zeros_(initial[-1].bias)
initial=initial.state_dict()
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,2]
reports={};strata={};norms={};ends={}
for arm in ('own','cross'):
 out=ROOT/f'outputs/pure_dl_sw_private_latent_v321_fold5_r1/{arm}';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
 assert (STUDY/f'{arm}_assembled.json').exists();reports[arm]=rep;ends[arm]=(out/'report.json').stat().st_mtime
 pred=y.copy();pred[:,2]=np.load(out/'channel_pred.npy')
 strata[arm]={n:float(base.metric(y[m],pred[m])['SW']) for n,m in [('default',dm),('continuous',~dm)]}
 state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms[arm]={n:float((state['latent_fuse.'+n].float()-v.float()).norm()) for n,v in initial.items()}
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Two overlapping jobs after48.126s stagger, both include donorforward/data/initial/train/bestrestore. Pair creation-to-lastterminal excludesprelaunch andfailedstartup/CPUfinalization. Not isolatedspeed comparison.')
timing['campaign_wall_seconds_including_failed_startup_fix']=max(ends.values())-timing['campaign_started_epoch_approx']
result=dict(reports=reports,strata=strata,best_latent_fusion_delta_norms=norms,cross_minus_own=reports['cross']['target_score']-reports['own']['target_score'],cross_minus_own_CONT=strata['cross']['continuous']-strata['own']['continuous'],gain_vs297={a:r['target_score']-74.16003022301975 for a,r in reports.items()},CONT_vs297={a:v['continuous']-17.502752767347182 for a,v in strata.items()},CONT_vs252={a:v['continuous']-17.88486813991017 for a,v in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,adopted=False,feature_rank_and_context_differ=True)
for n,v in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]: (STUDY/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(result))

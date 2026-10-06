"""CPU terminal late-fusion evidence, no launch or automatic adoption."""
from pathlib import Path
import sys,json
import torch,numpy as np
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
from torch import nn
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,2]
out=ROOT/'outputs/pure_dl_sw_late_latent_v323_fold5/late';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
assert (STUDY/'late_assembled.json').exists()
def strata(pred):
 p=y.copy();p[:,2]=pred
 return {n:float(base.metric(y[m],p[m])['SW']) for n,m in [('default',dm),('continuous',~dm)]}
full=strata(np.load(out/'channel_pred.npy'))
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(20261321);initial=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,48),nn.GELU(),nn.Linear(48,48));nn.init.zeros_(initial[-1].weight);nn.init.zeros_(initial[-1].bias)
state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float((state['latent_fuse.'+n].float()-v.float()).norm()) for n,v in initial.state_dict().items()}
paired=None
if (out/'paired22.json').exists():
 cap=read(out/'paired22.json');ss=strata(np.load(out/'paired22_best_pred.npy'))
 paired=dict(cap=cap,strata=ss,delta_whole_vs321_own=cap['target_score']-74.16603018049169,delta_CONT_vs321_own=ss['continuous']-17.911355427689944,scope='Predefined best through22 vs historical321own22; removednonfunctionalwork/auxgradientpaths limits remain')
result=dict(report=rep,strata=full,best_latent_fusion_delta_norms=norms,predefined_equal_epoch22=paired,full_absolute_whole_vs321_own=rep['target_score']-74.16603018049169,full_absolute_CONT_vs321_own=full['continuous']-17.911355427689944,full_best_after22_is_not_equal_work_gain=rep['best_epoch']>22,independent_disk_reload=False,adopted=False,complete_five_folds=False,remaining_folds_started=False,screening_fold=5)
if (STUDY/'TIMING.json').exists():
 timing=read(STUDY/'TIMING.json');timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(out/'report.json').stat().st_mtime-timing['started_unix'],interpretation='Includes prep/initial/train/bestrestore, concurrentdifferentPOR exploration disclosed; excludes prelaunch/CPUfinalization. Not isolatedspeed comparison versus321donorwork.')
 result['timing']=timing;(STUDY/'TIMING.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

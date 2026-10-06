"""CPU terminal actual evidence, no automatic adoption or launch."""
from pathlib import Path
import sys,json,torch,numpy as np
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
from sw_latent_morphology_v326 import Morphology
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
out=ROOT/'outputs/pure_dl_por_latent_morphology_v328_fold5/depth';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (STUDY/'depth_assembled.json').exists()
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,0]
def strata(pred):
 p=y.copy();p[:,0]=pred
 return {n:float(base.metric(y[m],p[m])['POR']) for n,m in [('default',dm),('continuous',~dm)]}
ss=strata(np.load(out/'channel_pred.npy'))
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(20261326);initial=Morphology('depth')
state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float((state['morphology.'+n].float()-v.float()).norm()) for n,v in initial.state_dict().items()}
result=dict(report=rep,strata=ss,actual_best_morphology_parameter_deltas=norms,whole_vs_historical_reference=rep['target_score']-78.03055279414464,CONT_vs_historical_reference=ss['continuous']-41.6750799070714,reference_whole=78.03055279414464,reference_CONT=41.6750799070714,independent_disk_reload=False,adopted=False,complete_five_folds=False,remaining_folds_started=False,screening_fold=5)
result['comparison_limits']='Historical302capacity76592 vs32882016; one depthscreen only, no puredepthcausalclaim. Freshchannelcontrol onlyifwholeCONTpositive.'

timing=read(STUDY/'TIMING.json');timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(out/'report.json').stat().st_mtime-timing['started_epoch'],interpretation='Actualconcurrencyrecorded; includesprep/initial/train/bestrestore, excludes implementation/CPUfinalization; notisolatedspeedranking')
result['timing']=timing;(STUDY/'TIMING.json').write_text(json.dumps(timing,indent=2),encoding='utf-8');(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

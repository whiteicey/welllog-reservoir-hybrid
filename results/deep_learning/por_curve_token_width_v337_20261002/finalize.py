"""Terminal CPU evidence for width128, no GPU/promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
o=R/'outputs/pure_dl_por_curve_token_width_v337_fold5/wide';rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/'wide_assembled.json').exists()
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,0]
def strata(pred):
 p=y.copy();p[:,0]=pred
 return {n:float(b.metric(y[sel],p[sel])['POR']) for n,sel in [('default',m),('continuous',~m)]}
ss=strata(np.load(o/'channel_pred.npy'));state=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float(v.float().norm()) for n,v in state.items() if n in ('curve_interaction.bridge.weight','curve_interaction.bridge.bias')}
cap=None
if (o/'paired42.json').exists():
 c=read(o/'paired42.json');c['strata']=strata(np.load(o/'paired42_best_pred.npy'));cap=c
result=dict(report=rep,strata=ss,bridge_zero_initialized_parameter_norms=norms,whole_vs335=rep['target_score']-78.04221263500206,CONT_vs335=ss['continuous']-41.724510830138236,predefined42=cap,stopped_before42=rep['stopped_epoch']<42,full_best_after42_not_equalwork=rep['best_epoch']>42,adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,comparison_limits='Historical128336/70768token64 vs330128/272560token128, initialization/weightshapes/compute differ; notcapacitymatchedcausal orrobustscalinggain')
timing=read(S/'TIMING.json');timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-timing['started_epoch']);result['timing']=timing;(S/'TIMING.json').write_text(json.dumps(timing,indent=2),encoding='utf-8');(S/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

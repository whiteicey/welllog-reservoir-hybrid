"""Terminal CPU evidence for conditionalRPS, no GPU/promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
o=R/'outputs/pure_dl_por_coupling_conditional_rps_v341_fold5/conditional';rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/'conditional_assembled.json').exists()
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,0]
def strata(pred):
 p=y.copy();p[:,0]=pred
 return {n:float(b.metric(y[sel],p[sel])['POR']) for n,sel in [('default',m),('continuous',~m)]}
ss=strata(np.load(o/'channel_pred.npy'));state=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float(v.float().norm()) for n,v in state.items() if n in ('curve_interaction.bridge.weight','curve_interaction.bridge.bias','curve_interaction.temporal.up.weight','curve_interaction.temporal.up.bias')}
cap=None
if (o/'paired42.json').exists():
 c=read(o/'paired42.json');c['strata']=strata(np.load(o/'paired42_best_pred.npy'));cap=c
result=dict(report=rep,strata=ss,bridge_zero_initialized_parameter_norms=norms,whole_vs338=rep['target_score']-78.0458220539967,CONT_vs338=ss['continuous']-41.73489002626708,CONT_vs_original151=ss['continuous']-41.95396105490406,predefined42=cap,stopped_before42=rep['stopped_epoch']<42,full_best_after42_not_equalwork=rep['best_epoch']>42,adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,comparison_limits='Same134752 architecture/init as historical338depth, globalrowmean conditionalRPS instead of joint; defaultdensitygradients removed and continuous effectivegradientweight changes via no frozen gate factor. Historical not concurrent replicate, no robustness proof.')
timing=read(S/'TIMING.json');timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-timing['started_epoch']);result['timing']=timing;(S/'TIMING.json').write_text(json.dumps(timing,indent=2),encoding='utf-8');(S/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

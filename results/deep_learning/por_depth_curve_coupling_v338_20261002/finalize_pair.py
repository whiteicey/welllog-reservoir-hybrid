"""Terminal paired CPU scoring/cost evidence; no GPU or promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=base.dmask(y)[:,0]
results={}
for arm in ('control','depth'):
 o=R/'outputs/pure_dl_por_depth_curve_coupling_v338_fold5'/arm;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/(arm+'_assembled.json')).exists()
 p=y.copy();p[:,0]=np.load(o/'channel_pred.npy');ss={name:float(base.metric(y[sel],p[sel])['POR']) for name,sel in [('default',m),('continuous',~m)]}
 state=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];router_norms={n:float(v.float().norm()) for n,v in state.items() if n in ('curve_interaction.temporal.up.weight','curve_interaction.temporal.up.bias')}
 timing=read(S/(arm+'_TIMING.json'));timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],terminal_epoch=(o/'report.json').stat().st_mtime,process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-timing['started_epoch']);(S/(arm+'_TIMING.json')).write_text(json.dumps(timing,indent=2),encoding='utf-8')
 cap=read(o/'paired42.json') if (o/'paired42.json').exists() else None
 if cap:
  cp=y.copy();cp[:,0]=np.load(o/'paired42_best_pred.npy');cap['continuous']=float(base.metric(y[~m],cp[~m])['POR'])
 results[arm]=dict(report=rep,strata=ss,zero_initialized_temporal_up_norm=router_norms,whole_vs335=rep['target_score']-78.04221263500206,CONT_vs335=ss['continuous']-41.724510830138236,CONT_vs_original151=ss['continuous']-41.95396105490406,predefined42=cap,stopped_before42=rep['stopped_epoch']<42,timing=timing,independent_disk_reload=False)
a=results['control'];b=results['depth'];result=dict(arms=results,depth_minus_control_whole=b['report']['target_score']-a['report']['target_score'],depth_minus_control_CONT=b['strata']['continuous']-a['strata']['continuous'],comparison_limits='Same134752 physicalparameters/init,17depth taps vs samepoint cyclic8latent under identical availability; effectivefunction/informationcapacity differ. Original151only/allfreshencoderreadoutjoint. Historical335 not concurrent repeat. Predefined42; laterbest not equalwork, actual stopped work recorded.',pair_wall_seconds=max(x['timing']['terminal_epoch'] for x in results.values())-min(x['timing']['started_epoch'] for x in results.values()),stagger_seconds=abs(a['timing']['started_epoch']-b['timing']['started_epoch']),adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False)
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

"""Terminal paired CPU scoring/cost evidence; no GPU or promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=base.dmask(y)[:,2]
results={}
for arm in ('identity','noise'):
 o=R/'outputs/pure_dl_sw_variance_dsu_v334_fold5'/arm;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/(arm+'_assembled.json')).exists()
 p=y.copy();p[:,2]=np.load(o/'channel_pred.npy');ss={name:float(base.metric(y[sel],p[sel])['SW']) for name,sel in [('default',m),('continuous',~m)]}
 state=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];router_norms={n:float(v.float().norm()) for n,v in state.items() if n in ('morphology.projection.weight','morphology.projection.bias')}
 timing=read(S/(arm+'_TIMING.json'));timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],terminal_epoch=(o/'report.json').stat().st_mtime,process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-timing['started_epoch']);(S/(arm+'_TIMING.json')).write_text(json.dumps(timing,indent=2),encoding='utf-8')
 cap=read(o/'paired27.json') if (o/'paired27.json').exists() else None
 results[arm]=dict(report=rep,strata=ss,zero_initialized_morph_projection_norm=router_norms,whole_vs326=rep['target_score']-74.1695430320749,CONT_vs326=ss['continuous']-17.96344208824835,predefined27=cap,stopped_before27=rep['stopped_epoch']<27,timing=timing,independent_disk_reload=False)
a=results['identity'];b=results['noise'];result=dict(arms=results,noise_minus_identity_whole=b['report']['target_score']-a['report']['target_score'],noise_minus_identity_CONT=b['strata']['continuous']-a['strata']['continuous'],comparison_limits='Same92289 parameters/init and identity/noise random draws; only training boundedvarianceDSU differs. Allfreshencoder/readoutjoint, original104only. Predeclared27 control snapshots exist ifreached; laterbest not equalwork evidence. Historical326 not simultaneous repeat; traininputstats uncertainty not proven domainuncertainty.',pair_wall_seconds=max(x['timing']['terminal_epoch'] for x in results.values())-min(x['timing']['started_epoch'] for x in results.values()),stagger_seconds=abs(a['timing']['started_epoch']-b['timing']['started_epoch']),adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False)
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

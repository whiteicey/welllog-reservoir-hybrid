"""Terminal paired CPU analysis, no GPU or checkpoint selection by CONT."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
torch.set_num_threads(1);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,2]
def strata(v):
 p=y.copy();p[:,2]=v
 return {n:float(b.metric(y[sel],p[sel])['SW']) for n,sel in [('default',m),('continuous',~m)]}
arms={}
for stage in ('full','grouped'):
 o=R/'outputs/pure_dl_sw_curve_interaction_v344_fold5'/stage;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/f'{stage}_assembled.json').exists()
 ss=strata(np.load(o/'channel_pred.npy'));prefix=None
 if (o/'paired27.json').exists():prefix=read(o/'paired27.json');prefix['strata']=strata(np.load(o/'paired27_best_pred.npy'))
 tm=read(S/f'{stage}_TIMING.json');tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/f'{stage}_TIMING.json').write_text(json.dumps(tm,indent=2),encoding='utf-8')
 arms[stage]=dict(report=rep,strata=ss,whole_vs326=rep['target_score']-74.1695430320749,CONT_vs326=ss['continuous']-17.96344208824835,CONT_vs104=ss['continuous']-17.631889060480088,predefined27=prefix,full_best_after27_not_equalwork=rep['best_epoch']>27,timing=tm)
result=dict(arms=arms,full_minus_grouped_whole=arms['full']['report']['target_score']-arms['grouped']['report']['target_score'],full_minus_grouped_CONT=arms['full']['strata']['continuous']-arms['grouped']['strata']['continuous'],comparison_limits='Same physical111313 parameters/init, different group communication and effective function capacity. Historical32692289 not capacity matched, no causal attribution from that alone. Single seed, no robustness proof.',adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False)
result['pair_wall_process_seconds']=max((R/'outputs/pure_dl_sw_curve_interaction_v344_fold5'/a/'report.json').stat().st_mtime for a in arms)-min(v['timing']['started_epoch'] for v in arms.values());result['launch_stagger_seconds']=abs(arms['full']['timing']['started_epoch']-arms['grouped']['timing']['started_epoch']);result['pair_timing_limits']='Process start through last report includes explicit serial launch gaps; individual includes prep/warmup/restoration, not isolated architecture speed ranking'
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

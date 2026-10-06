"""Terminal sparse-reader paired CPU evidence, no GPU/promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
torch.set_num_threads(1);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,0]
def strata(v):
 p=y.copy();p[:,0]=v
 return {n:float(b.metric(y[sel],p[sel])['POR']) for n,sel in [('default',m),('continuous',~m)]}
arms={}
for stage in ('adaptive','fixed'):
 o=R/'outputs/pure_dl_por_adaptive_depth_read_v345_fold5'/stage;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/f'{stage}_assembled.json').exists()
 ss=strata(np.load(o/'channel_pred.npy'));prefix=None
 if (o/'paired42.json').exists():prefix=read(o/'paired42.json');prefix['strata']=strata(np.load(o/'paired42_best_pred.npy'))
 state=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float(v.float().norm()) for n,v in state.items() if n.startswith('curve_interaction.reader.')}
 selection=state['curve_interaction.reader.select.weight'];bias=state['curve_interaction.reader.select.bias'];selector_norms=dict(first5_weight=float(selection[:5].float().norm()),first5_bias=float(bias[:5].float().norm()),last5_weight=float(selection[5:].float().norm()),last5_bias=float(bias[5:].float().norm()),first5_use='offsets' if stage=='adaptive' else 'contentweights average, redundant antisymmetric directions')
 tm=read(S/f'{stage}_TIMING.json');tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/f'{stage}_TIMING.json').write_text(json.dumps(tm,indent=2),encoding='utf-8')
 arms[stage]=dict(report=rep,strata=ss,whole_vs341=rep['target_score']-78.04643071793232,CONT_vs341=ss['continuous']-41.736640291995144,CONT_vs151=ss['continuous']-41.95396105490406,predefined42=prefix,full_best_after42_not_equalwork=rep['best_epoch']>42,reader_norms=norms,selector_norms=selector_norms,timing=tm)
result=dict(arms=arms,adaptive_minus_fixed_whole=arms['adaptive']['report']['target_score']-arms['fixed']['report']['target_score'],adaptive_minus_fixed_CONT=arms['adaptive']['strata']['continuous']-arms['fixed']['strata']['continuous'],comparison_limits='Same136626 physicalparams/init but offset-vs-contentweight roles and effective function capacity differ. Historical341134752 not matched capacity. Single seed historical microgain not robustness.',adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False)
result['pair_wall_process_seconds']=max((R/'outputs/pure_dl_por_adaptive_depth_read_v345_fold5'/a/'report.json').stat().st_mtime for a in arms)-min(v['timing']['started_epoch'] for v in arms.values());result['launch_stagger_seconds']=abs(arms['adaptive']['timing']['started_epoch']-arms['fixed']['timing']['started_epoch']);result['pair_timing_limits']='Process start through last report includes explicit serial launch gaps; individual includes prep/warmup/restoration, not isolated architecture speed ranking'
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

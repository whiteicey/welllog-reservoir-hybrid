"""Terminal ordered-knot paired CPU evidence, no GPU/promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
torch.set_num_threads(1);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,1]
def strata(v):
 p=y.copy();p[:,1]=v
 return {n:float(b.metric(y[sel],p[sel])['PERM']) for n,sel in [('default',m),('continuous',~m)]}
arms={}
for stage in ('learned','fixed'):
 o=R/'outputs/pure_dl_perm_ordered_knots_v346_fold5'/stage;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/f'{stage}_assembled.json').exists()
 ss=strata(np.load(o/'channel_pred.npy'));prefix=None
 if (o/'paired19.json').exists():prefix=read(o/'paired19.json');prefix['strata']=strata(np.load(o/'paired19_best_pred.npy'))
 ck=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False);state=ck['state_dict'];logits=state['numeric.width_logits'].float();bounds=state['numeric.bounds'].float();eps=1/512;fractions=eps+(1-8*eps)*logits.softmax(-1);widths=(bounds[:,1]-bounds[:,0])[:,None]*fractions;left=bounds[:,:1]+torch.cat((torch.zeros_like(widths[:,:1]),widths.cumsum(-1)),dim=-1)
 diag=dict(width_logit_norm=float(logits.norm()),width_min_fraction=float(fractions.min()),width_max_fraction=float(fractions.max()),ordered_knots=left.tolist(),all_widths_positive=bool((widths>0).all()),projection_norm=float(state['numeric.projection.weight'].float().norm()),active_numeric_dimensions=logits.numel(),actual_trainable_parameters=ck['config']['trainable_parameters'],numeric_fit_info=ck['config']['initialization']['numeric_fit_info'],fixed_initial_logits_remain_zero=bool((logits==0).all()) if stage=='fixed' else None)
 tm=read(S/f'{stage}_TIMING.json');tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/f'{stage}_TIMING.json').write_text(json.dumps(tm,indent=2),encoding='utf-8')
 arms[stage]=dict(report=rep,strata=ss,whole_vs336=rep['target_score']-89.51838939879066,CONT_vs336=ss['continuous']-74.00963708319706,CONT_vs154=ss['continuous']-74.09691194381655,predefined19=prefix,full_best_after19_not_equalwork=rep['best_epoch']>19,numeric_diagnostics=diag,timing=tm)
result=dict(arms=arms,learned_minus_fixed_whole=arms['learned']['report']['target_score']-arms['fixed']['report']['target_score'],learned_minus_fixed_CONT=arms['learned']['strata']['continuous']-arms['fixed']['strata']['continuous'],comparison_limits='Same uniform initialknots/projection/baseparameter init, learned addsS<=72 trainable logitparameters, not matched trainablecapacity. Historical336 lacksbasis andcapacitydiffers. Single seed notrobustness.',adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False)
result['pair_wall_process_seconds']=max((R/'outputs/pure_dl_perm_ordered_knots_v346_fold5'/a/'report.json').stat().st_mtime for a in arms)-min(v['timing']['started_epoch'] for v in arms.values());result['launch_stagger_seconds']=abs(arms['learned']['timing']['started_epoch']-arms['fixed']['timing']['started_epoch']);result['pair_timing_limits']='Process start through last report includes explicit serial launch gaps; individual includes prep/warmup/restoration, not isolated architecture speed ranking'
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

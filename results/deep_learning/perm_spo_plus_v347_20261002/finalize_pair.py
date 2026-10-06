"""Terminal matched SPO+ paired CPU evidence, no GPU/promotion."""
from pathlib import Path
import sys,json,torch,numpy as np
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
torch.set_num_threads(1);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);m=b.dmask(y)[:,1]
def strata(v):
 p=y.copy();p[:,1]=v
 return {n:float(b.metric(y[sel],p[sel])['PERM']) for n,sel in [('default',m),('continuous',~m)]}
arms={}
for stage in ('reference','spo'):
 o=R/'outputs/pure_dl_perm_spo_plus_v347_fold5'/stage;rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact'];assert (S/f'{stage}_assembled.json').exists()
 ss=strata(np.load(o/'channel_pred.npy'));prefix=None
 if (o/'paired19.json').exists():prefix=read(o/'paired19.json');prefix['strata']=strata(np.load(o/'paired19_best_pred.npy'))
 cfg=read(o/'config.json');h=read(o/'validation_history.json');diag=dict(actual_trainable_parameters=cfg['trainable_parameters'],SPO_coefficient=cfg['initialization']['spo_coefficient'],action_count=cfg['initialization']['decision_actions'],training_truth_rows=cfg['initialization']['actual_train_truth_rows'],last_SPO_loss=h[-1]['spo_loss'],last_loss_augmented_disagreement=h[-1]['loss_augmented_disagreement'],own_whole_best_epoch=rep['best_epoch'],SPO_loss_is_not_official_regret=True)
 tm=read(S/f'{stage}_TIMING.json');tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/f'{stage}_TIMING.json').write_text(json.dumps(tm,indent=2),encoding='utf-8')
 arms[stage]=dict(report=rep,strata=ss,whole_vs336=rep['target_score']-89.51838939879066,CONT_vs336=ss['continuous']-74.00963708319706,CONT_vs154=ss['continuous']-74.09691194381655,predefined19=prefix,full_best_after19_not_equalwork=rep['best_epoch']>19,decision_loss_diagnostics=diag,timing=tm)
result=dict(arms=arms,spo_minus_reference_whole=arms['spo']['report']['target_score']-arms['reference']['report']['target_score'],spo_minus_reference_CONT=arms['spo']['strata']['continuous']-arms['reference']['strata']['continuous'],comparison_limits='Same37386 parameters/init, sameactionkernel/decoder, onlyfixedSPO coefficient0 vs.1; freshreference computesadditionalzero-coefficient term for faircompute/RNG. Historical336samecapacity not concurrent repeat, single seed not robustness.',adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False)
result['pair_wall_process_seconds']=max((R/'outputs/pure_dl_perm_spo_plus_v347_fold5'/a/'report.json').stat().st_mtime for a in arms)-min(v['timing']['started_epoch'] for v in arms.values());result['launch_stagger_seconds']=abs(arms['reference']['timing']['started_epoch']-arms['spo']['timing']['started_epoch']);result['pair_timing_limits']='Process start through last report includes explicit serial launch gaps; individual includes prep/warmup/restoration, no-original336-speedranking because bothcomputeadditionaldiagnostics'
(S/'PAIRED_FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

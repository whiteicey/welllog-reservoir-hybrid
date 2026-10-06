"""Terminal CPU evidence only; no GPU/promotion/restart."""
from pathlib import Path
import argparse,json,sys,torch,numpy as np,psutil
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
a=argparse.ArgumentParser();a.add_argument('--stage',choices=['depth','control'],required=True);arm=a.parse_args().stage
O=R/'outputs/pure_dl_perm_depth_path_residual_v359_fold5'/arm;state=read(O/'runtime_status.json')
try:live=any(Path(v).name=='train_perm_depth_path_residual_v359.py' for v in psutil.Process(state['pid']).cmdline())
except psutil.Error:live=False
assert not live and (O/'report.json').exists(),'Terminal real process/report required'
rep=read(O/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);pred=np.load(O/'channel_pred.npy');p=y.copy();p[:,1]=pred;m=b.dmask(y)[:,1]
ss={n:float(b.metric(y[ix],p[ix])['PERM']) for n,ix in [('default',m),('continuous',~m)]}
combined=np.column_stack((np.load(R/'outputs/pure_dl_por_coupling_conditional_rps_v341_fold5/conditional/channel_pred.npy'),pred,np.load(R/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5/aligned/channel_pred.npy')));scores=b.metric(y,combined)
np.save(S/(arm+'_diagnostic_eight_stage_pred.npy'),combined)
ck=torch.load(O/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float(v.float().norm()) for n,v in ck.items() if n.startswith(('path_residual.bridge.','temporal_residual.up.'))}
timing=read(S/(arm+'_TIMING.json'));timing.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(O/'report.json').stat().st_mtime-timing['started_epoch']);(S/(arm+'_TIMING.json')).write_text(json.dumps(timing,indent=2),encoding='utf-8')
f=dict(report=rep,strata=ss,whole_vs358area=rep['target_score']-89.51746356358993,CONT_vs358area=ss['continuous']-74.01331830014448,whole_vs336=rep['target_score']-89.51838939879066,CONT_vs336=ss['continuous']-74.00963708319706,CONT_vs_native154=ss['continuous']-74.09691194381553,diagnostic_eight_stage_scores=scores,TOTAL_vs_retained351=scores['TOTAL']-80.70612972323806,path_bridge_norms=norms,timing=timing,adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,strict_808_passed=scores['TOTAL']>80.8,comparison_limits='45274params vs35839162 and33637386 historical/nonconcurrent; learneddepth branch extra6112, controlnottrained/no matchedgainclaim. Single seed/wholebest/no CONToracle.')
(S/(arm.upper()+'_TERMINAL.json')).write_text(json.dumps(f,indent=2),encoding='utf-8')
t='\nV359 '+arm+' terminal PERM'+str(rep['target_score'])+'/CONT'+str(ss['continuous'])+', vs336whole'+str(f['whole_vs336'])+'/CONT'+str(f['CONT_vs336'])+', best/stop'+str(rep['best_epoch'])+'/'+str(rep['stopped_epoch'])+', effectivepatience'+str(rep['effective_patience'])+'/SOTAbroken'+str(rep['SOTA_broken'])+'. Actualdiagnostic8TOTAL'+str(scores['TOTAL'])+'; notadopted beforepaireddecision. '+str(rep['seconds'])+'s solo/'+str(rep['peak_MiB'])+'MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.\n'
for pth in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md']:
 with pth.open('a',encoding='utf-8') as h:h.write(t)
print(json.dumps(f))

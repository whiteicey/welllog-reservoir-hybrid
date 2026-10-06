"""Terminal spectral-arm CPU evidence; no GPU or automatic promotion."""
from pathlib import Path
import sys,json,argparse,torch,numpy as np,psutil
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['aligned','permuted'],required=True);a=ap.parse_args();arm=a.stage
o=R/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5'/arm;state=read(o/'runtime_status.json')
try:live=any(Path(v).name=='train_sw_dynamic_cross_spectrum_v351.py' for v in psutil.Process(state['pid']).cmdline())
except psutil.Error:live=False
assert not live and (o/'report.json').exists(),'Require terminal process and report'
rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);pred=np.load(o/'channel_pred.npy');p=y.copy();p[:,2]=pred;m=b.dmask(y)[:,2]
ss={n:float(b.metric(y[ix],p[ix])['SW']) for n,ix in [('default',m),('continuous',~m)]}
combined=np.column_stack((np.load(R/'outputs/pure_dl_por_coupling_conditional_rps_v341_fold5/conditional/channel_pred.npy'),np.load(R/'outputs/pure_dl_perm_low_rank_cross_v336_fold5/cross/channel_pred.npy'),pred));scores=b.metric(y,combined)
np.save(S/(arm+'_diagnostic_eight_stage_pred.npy'),combined)
ck=torch.load(o/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms={n:float(v.float().norm()) for n,v in ck.items() if n.startswith('cross_spectrum.') and v.is_floating_point()}
perm=ck['cross_spectrum.donor_permutation'].numpy();assert np.array_equal(np.sort(perm),np.arange(256))
tm=read(S/(arm+'_TIMING.json'));tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/(arm+'_TIMING.json')).write_text(json.dumps(tm,indent=2),encoding='utf-8')
prefix=read(o/'paired27.json') if (o/'paired27.json').exists() else None
f=dict(report=rep,strata=ss,whole_vs326=rep['target_score']-74.1695430320749,CONT_vs326=ss['continuous']-17.96344208824835,CONT_vs_native104=ss['continuous']-17.631889060480088,diagnostic_eight_stage_scores=scores,TOTAL_vs_retained341=scores['TOTAL']-80.70470556618264,spectrum_parameter_norms=norms,timing=tm,predefined27=prefix,adopted=False,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,strict_808_passed=scores['TOTAL']>80.8,comparison_limits='Same92841params/init between aligned/permuted but function class/spectral smoothness differ; historical326capacity92289 differs/nonconcurrent; single seed and whole-selected strata, no CONToracle. Norms not importance proof.')
(S/(arm.upper()+'_TERMINAL.json')).write_text(json.dumps(f,indent=2),encoding='utf-8')
t='\nV351 '+arm+' terminal SW'+str(rep['target_score'])+'/CONT'+str(ss['continuous'])+', vs326whole'+str(f['whole_vs326'])+'/CONT'+str(f['CONT_vs326'])+', best/stop'+str(rep['best_epoch'])+'/'+str(rep['stopped_epoch'])+'. Diagnostic8TOTAL'+str(scores['TOTAL'])+' (notadopted untilpair); spectrum mixing/bridgenorms recorded/'+str(rep['seconds'])+'s solo/'+str(rep['peak_MiB'])+'MiB inclprep/warmup/restoration;27prefix/'+str(bool(prefix))+'/144665RAMbestreplay/frozenexact/notdiskreload. Strict808='+str(f['strict_808_passed'])+'/fullOOF82unmet/nootherfolds.\n'
for pth in (S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md'):
 with pth.open('a',encoding='utf-8') as h:h.write(t)
print(json.dumps(f))

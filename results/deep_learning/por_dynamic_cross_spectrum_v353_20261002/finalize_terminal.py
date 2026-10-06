"""Terminal CPU evidence only; no GPU, promotion or deletion."""
from pathlib import Path
import sys,json,torch,numpy as np,psutil
S=Path(__file__).resolve().parent;R=S.parents[2];sys.path.insert(0,str(R/'experiments/pure_dl'));import train_invariant as b
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
o=R/'outputs/pure_dl_por_dynamic_cross_spectrum_v353_fold5/aligned';state=read(o/'runtime_status.json')
try:
 live=any(Path(a).name=='train_por_dynamic_cross_spectrum_v353.py' for a in psutil.Process(state['pid']).cmdline())
except psutil.Error:live=False
assert not live and (o/'report.json').exists(),'Require terminal real process and report'
rep=read(o/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
d=torch.load(R/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(d['truth'][d['validation']]);p=y.copy();pred=np.load(o/'channel_pred.npy');p[:,0]=pred;m=b.dmask(y)[:,0]
strata={n:float(b.metric(y[ix],p[ix])['POR']) for n,ix in [('default',m),('continuous',~m)]}
combined=np.column_stack((pred,np.load(R/'outputs/pure_dl_perm_low_rank_cross_v336_fold5/cross/channel_pred.npy'),np.load(R/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5/aligned/channel_pred.npy')))
score=b.metric(y,combined);np.save(S/'diagnostic_eight_stage_pred.npy',combined)
tm=read(S/'aligned_TIMING.json');tm.update(terminal=True,individual_seconds=rep['seconds'],peak_MiB=rep['peak_MiB'],process_creation_to_terminal_seconds=(o/'report.json').stat().st_mtime-tm['started_epoch']);(S/'aligned_TIMING.json').write_text(json.dumps(tm,indent=2),encoding='utf-8')
prefix=read(o/'paired42.json') if (o/'paired42.json').exists() else None
f=dict(report=rep,strata=strata,whole_vs_serial345=rep['target_score']-78.04529023876282,CONT_vs_serial345=strata['continuous']-41.73733625647294,whole_vs341=rep['target_score']-78.04643071793232,CONT_vs341=strata['continuous']-41.736640291995144,CONT_vs_native151=strata['continuous']-41.95396105490406,diagnostic_eight_stage_scores=score,TOTAL_vs_retained351=score['TOTAL']-80.70612972323806,timing=tm,predefined42=prefix,adopted=False,screening_fold=5,complete_five_folds=False,independent_disk_reload=False,remaining_folds_started=False,permuted_control_launched=False,comparison_limits='135304params vs341134752 historical/nonconcurrent; extra552capacity/transfermodule, no matchedpermuted claim. Wholecheckpoint selected/singleseed/no CONToracle. SW351positive remains separate.')
f['adopted']=False
f['closed_setting']='POR351 amplitude-spectrum transfer below retained341whole; permuted not trained/no matchedgain claim, close transfer configuration no grid; SW351positive preserved'
(S/'FINAL.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
t='\nV353 terminal POR'+str(rep['target_score'])+'/CONT'+str(strata['continuous'])+', best/stop'+str(rep['best_epoch'])+'/'+str(rep['stopped_epoch'])+', diagnostic vsserial345 whole'+str(f['whole_vs_serial345'])+'/CONT'+str(f['CONT_vs_serial345'])+', vs341whole'+str(f['whole_vs341'])+'. Diagnostic8TOTAL'+str(score['TOTAL'])+'; adopted='+str(f['adopted'])+'. '+f['closed_setting']+'. Permutedarm notlaunched/no matchedgainclaim. Solo individual'+str(rep['seconds'])+'s/peak'+str(rep['peak_MiB'])+'MiB includesprep/warmup/restoration;144665RAMbestreplay/frozenexact/notdiskreload/fullOOF82unmet/nootherfolds.\n'
for pth in (S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md'):
 with pth.open('a',encoding='utf-8') as h:h.write(t)
print(json.dumps(f))

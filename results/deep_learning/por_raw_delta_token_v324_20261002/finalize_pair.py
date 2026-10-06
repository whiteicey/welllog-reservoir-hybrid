"""Completed CPU paired evidence, never launches or auto-promotes."""
from pathlib import Path
import sys,json
import numpy as np,torch
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False);y=np.asarray(data['truth'][data['validation']]);dm=base.dmask(y)[:,0]
reports={};strata={};norms={};ends={}
for arm in ('adaptive','reference'):
 out=ROOT/f'outputs/pure_dl_por_raw_delta_token_v324_fold5/{arm}';rep=read(out/'report.json');assert rep['predictions_reproduced_exactly'] and rep['frozen_parameters_exact']
 assert (STUDY/f'{arm}_assembled.json').exists();reports[arm]=rep;ends[arm]=(out/'report.json').stat().st_mtime
 pred=y.copy();pred[:,0]=np.load(out/'channel_pred.npy');strata[arm]={n:float(base.metric(y[m],pred[m])['POR']) for n,m in [('default',dm),('continuous',~dm)]}
 state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict'];norms[arm]={n:float(state['window_norm.'+n].float().norm()) for n in ('shift','scale')}
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Actual launch stagger and concurrency recorded in this study. Includes data/initial/train/bestrestore; pair process-to-lastterminal excludesprelaunch/CPUfinalization. No isolatedspeedranking.')
result=dict(reports=reports,strata=strata,best_normalizer_delta_norms=norms,adaptive_minus_reference=reports['adaptive']['target_score']-reports['reference']['target_score'],adaptive_minus_reference_CONT=strata['adaptive']['continuous']-strata['reference']['continuous'],gain_vs302={a:r['target_score']-78.03055279414464 for a,r in reports.items()},CONT_vs302={a:v['continuous']-41.6750799070714 for a,v in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,adopted=False)
for n,v in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]: (STUDY/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(result))

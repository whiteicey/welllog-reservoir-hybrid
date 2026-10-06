"""CPU-only completed whole-target evidence; never starts training or promotes."""
import json,sys
from pathlib import Path
import numpy as np
import torch
STUDY=Path(__file__).resolve().parent;ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,1]
reports={};strata={};norms={};ends={}
for arm in ('rational','polynomial'):
    out=ROOT/f'outputs/pure_dl_perm_group_shape_v312_fold5/{arm}'
    report=read(out/'report.json')
    assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=report;ends[arm]=(out/'report.json').stat().st_mtime
    pred=truth.copy();pred[:,1]=np.load(out/'channel_pred.npy')
    strata[arm]={n:float(base.metric(truth[m],pred[m])['PERM']) for n,m in [('default',default),('continuous',~default)]}
    state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict']
    norms[arm]={n:float((t.float()-state[n.rsplit('.',1)[0]+'.anchor_'+n.rsplit('.',1)[1]].float()).norm()) for n,t in state.items() if n in ('body_shape.a','body_shape.b','readout_shape.a','readout_shape.b')}
timing=read(STUDY/'BATCH_TIMING.json')
timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Sequential launches, rational overlaps311shuffled and polynomial mostly runs alone; wall includes launch gap and all environment elapsed work inside interval, excludes preparation beforefirstlaunch and CPUfinalization afterward. Not an isolated serial-versus-parallel speed ranking.')
result=dict(reports=reports,strata=strata,best_shape_delta_norms=norms,rational_minus_polynomial=reports['rational']['target_score']-reports['polynomial']['target_score'],rational_minus_polynomial_CONT=strata['rational']['continuous']-strata['polynomial']['continuous'],gain_vs_retained291={a:r['target_score']-89.5155547932572 for a,r in reports.items()},CONT_vs_retained291={a:s['continuous']-74.0010663397606 for a,s in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,automatic_adoption=False)
for name,value in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\nFinal312 whole scores { {a:r['target_score'] for a,r in reports.items()} }; rational-polynomial {result['rational_minus_polynomial']:+.12f}; CONT paired {result['rational_minus_polynomial_CONT']:+.12f}; strata {strata}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall {timing['same_work_pair_wall_seconds']:.3f}s; no fullOOF or remaining folds.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(result))

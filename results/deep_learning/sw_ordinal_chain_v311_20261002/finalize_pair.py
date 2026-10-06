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
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,2]
reports={};strata={};norms={};ends={}
for arm in ('ordered','shuffled'):
    out=ROOT/f'outputs/pure_dl_sw_ordinal_chain_v311_fold5/{arm}'
    report=read(out/'report.json')
    assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=report;ends[arm]=(out/'report.json').stat().st_mtime
    pred=truth.copy();pred[:,2]=np.load(out/'channel_pred.npy')
    strata[arm]={n:float(base.metric(truth[m],pred[m])['SW']) for n,m in [('default',default),('continuous',~default)]}
    state=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict']
    norms[arm]={n:float(t.float().norm()) for n,t in state.items() if n.startswith('readout.linear.')}
timing=read(STUDY/'BATCH_TIMING.json')
timing.update(terminal=True,same_work_pair_wall_seconds=max(ends.values())-timing['started_epoch'],individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},interpretation='Two concurrently started arms, different stopping times. Includes data/initial/train/best restore; excludes code preparation and CPU record. Not a serial speed ranking.')
result=dict(reports=reports,strata=strata,best_readout_norms=norms,ordered_minus_shuffled=reports['ordered']['target_score']-reports['shuffled']['target_score'],ordered_minus_shuffled_CONT=strata['ordered']['continuous']-strata['shuffled']['continuous'],gain_vs_retained297={a:r['target_score']-74.16003022301975 for a,r in reports.items()},CONT_vs_retained297={a:s['continuous']-17.502752767347182 for a,s in strata.items()},CONT_vs_strong252={a:s['continuous']-17.88486813991017 for a,s in strata.items()},timing=timing,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,independent_disk_reload=False,automatic_adoption=False)
for name,value in [('PAIRED_FINAL.json',result),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\nFinal311 whole scores { {a:r['target_score'] for a,r in reports.items()} }; ordered-shuffled {result['ordered_minus_shuffled']:+.12f}; CONT paired {result['ordered_minus_shuffled_CONT']:+.12f}; strata {strata}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall {timing['same_work_pair_wall_seconds']:.3f}s; no fullOOF or remaining folds.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(result))

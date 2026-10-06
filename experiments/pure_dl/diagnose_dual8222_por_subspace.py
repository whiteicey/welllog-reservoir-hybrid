"""CPU-only well-cluster resampling and timing; exploratory, selection unadjusted."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_por_subspace_20260928'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    progress=json.loads((STUDY/'progress.json').read_text(encoding='utf-8'))
    assert progress['all_complete'],'Final reports required'
    draws=np.random.default_rng(20260928).integers(0,16,size=(10000,16))
    rows={};times=[]
    for r in progress['runs']:
        d=r['diagnostic'];f=r['final_report']
        wells=sorted(d['per_well'],key=lambda w:w['well_index'])
        assert len(wells)==16
        n=np.array([w['n'] for w in wells],dtype=float)
        delta=np.array([w['TOTAL_delta'] for w in wells])
        samples=(n[draws]*delta[draws]).sum(1)/n[draws].sum(1)
        folder=ROOT/'outputs'/r['id']
        history=json.loads((folder/'fold_5_validation_history.json').read_text(encoding='utf-8'))
        launch=json.loads((folder/'launch.json').read_text(encoding='utf-8'))
        start=datetime.fromisoformat(launch['started_at']);end=datetime.fromisoformat(f['finished_at'])
        times.append((start,end))
        best=next((v for v in history if v['epoch']==f['best_epoch']),None)
        rows[r['mode']]={'official_delta':d['delta_initial'],'equal_well_mean_delta':float(delta.mean()),
            'exploratory_well_percentile_95':np.quantile(samples,[.025,.975]).tolist(),
            'wins':d['well_wins'],'losses':d['well_losses'],
            'training_NLL_at_best':best['por_continuous_nll'] if best else None,
            'training_NLL_at_last':history[-1]['por_continuous_nll'],
            'launch_to_finish_seconds':(end-start).total_seconds(),'reported_process_seconds':f['seconds']}
    first=min(x[0] for x in times);last=max(x[1] for x in times)
    active=0;maximum=0
    for _,change in sorted([(a,1) for a,b in times]+[(b,-1) for a,b in times],key=lambda x:(x[0],x[1])):
        active+=change;maximum=max(maximum,active)
    result={'observed_at':datetime.now().astimezone().isoformat(),'runs':rows,
        'timing':{'first_launch':first.isoformat(),'last_completion':last.isoformat(),
                  'batch_span_seconds':(last-first).total_seconds(),'maximum_training_processes':maximum,
                  'scope':'first launch to last completion; includes setup/concurrency and explicit launch gaps; not a speed benchmark'},
        'limitations':'10000 whole-well resamples,16 wells,length weighting retained. Selection UNADJUSTED after repeated fold5 use; exploratory spread, not confirmatory independent-test confidence.'}
    (STUDY/'well_uncertainty_and_timing.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

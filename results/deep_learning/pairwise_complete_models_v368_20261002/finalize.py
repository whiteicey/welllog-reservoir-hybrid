"""CPU only: score every pair and actually assemble selected fixed head ratios."""
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask

def main():
    if (STUDY/'FINAL.json').exists():raise RuntimeError('Completed diagnostic immutable')
    started=time.perf_counter();plan=json.loads((STUDY/'PLAN.json').read_text());targets=('POR','PERM','SW')
    if (STUDY/'AUGMENT_PLAN.json').exists():
        extension=json.loads((STUDY/'AUGMENT_PLAN.json').read_text())
        plan['models']=extension['models'];plan['pairs']+=extension['pairs']
    y=np.load(STUDY/'truth.npy');assert y.shape==(144665,3)
    mask=dmask(y);grid={};gpu_reports=[]
    for c,t in enumerate(targets):
        r=json.loads((STUDY/(t+'_GPU_FINAL.json')).read_text())
        assert r['coverage_exactly_once'] and r['native_disk_replay_exact'] and r['endpoints_exact']
        gpu_reports.append(r)
        with np.load(STUDY/(t+'_grids.npz')) as z:
            actions=z['actions'];weights=z['weights'];indices=z['indices']
            specs={m['id']:m for m in plan['models']}
            for mi,ident in enumerate(r['models']):
                native_pred=y.copy();native_pred[:,c]=z['native'][:,mi]
                assert abs(metric(y,native_pred)[t]-specs[ident]['scores'][t])<1e-9
            for pi,p in enumerate(r['pairs']):
                pred=y.copy();scores=[]
                for wi in range(101):
                    pred[:,c]=actions[indices[pi,:,wi]];scores.append(metric(y,pred)[t])
                grid[p['id'],t]=(weights.copy(),np.array(scores),actions[indices[pi]].copy())
        old=ROOT/plan['reused_cache']
        with np.load(old/(t+'_grid_actions.npz')) as z:
            report=json.loads((old/(t+'_RESULT.json')).read_text())
            assert report['endpoints_and_half_exact'] and report['old_disk_replay_exact'] and report['new_disk_replay_exact']
            grid[plan['reused_pair'],t]=(z['weights'].copy(),np.array([v['score'] for v in report['scores_by_weight']]),z['actions'][z['indices']].copy())
        if (STUDY/'AUGMENT_PLAN.json').exists():
            r=json.loads((STUDY/('EXTRA_'+t+'_GPU_FINAL.json')).read_text())
            assert r['coverage_exactly_once'] and r['native_disk_replay_exact'] and r['endpoints_exact']
            gpu_reports.append(r)
            with np.load(STUDY/('EXTRA_'+t+'_grids.npz')) as z:
                actions=z['actions'];weights=z['weights'];indices=z['indices']
                for mi,ident in enumerate(r['models']):
                    native_pred=y.copy();native_pred[:,c]=z['native'][:,mi]
                    assert abs(metric(y,native_pred)[t]-specs[ident]['scores'][t])<1e-9
                for pi,p in enumerate(r['pairs']):
                    pred=y.copy();scores=[]
                    for wi in range(101):
                        pred[:,c]=actions[indices[pi,:,wi]];scores.append(metric(y,pred)[t])
                    grid[p['id'],t]=(weights.copy(),np.array(scores),actions[indices[pi]].copy())
    results=[];allrows=[];coeff=np.array([.3,.35,.35])
    for pair in plan['pairs']:
        selected={};cols=[];grids=[];weights=None
        for t in targets:
            w,s,pred=grid[pair['id'],t]
            i=max(range(101),key=lambda k:(s[k],w[k]))
            selected[t]=dict(weight_A=float(w[i]),weight_B=float(1-w[i]),score=float(s[i]),grid_index=i)
            cols.append(pred[:,i]);grids.append(s);weights=w
            for wi in range(101):allrows.append(dict(pair=pair['id'],target=t,weight_A=float(w[wi]),score=float(s[wi])))
        pred=np.column_stack(cols);assert pred.shape==y.shape and np.isfinite(pred).all()
        scores=metric(y,pred)
        assert all(abs(scores[t]-selected[t]['score'])<1e-9 for t in targets)
        totals=coeff@np.stack(grids);i=max(range(101),key=lambda k:(totals[k],weights[k]))
        common=metric(y,np.column_stack([grid[pair['id'],t][2][:,i] for t in targets]))
        equal=metric(y,np.column_stack([grid[pair['id'],t][2][:,50] for t in targets]))
        assert abs(common['TOTAL']-totals[i])<1e-9
        strata={t:{label:dict(rows=int(m.sum()),score=metric(y[m],pred[m])[t]) for label,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]} for c,t in enumerate(targets)}
        path=STUDY/(pair['id']+'_prediction.npy');np.save(path,pred)
        assert np.array_equal(np.load(path),pred)
        result=dict(**pair,scores=scores,selected_weights=selected,equal_weight_scores=equal,
                    common_weight=dict(weight_A=float(weights[i]),weight_B=float(1-weights[i]),scores=common),strata=strata,
                    strictly_above_8074=scores['TOTAL']>80.74,delta_vs_8074=scores['TOTAL']-80.74,
                    delta_vs_prior_per_head=scores['TOTAL']-plan['prior_per_head_reference'],
                    prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validation_rows=len(y),actual_three_head_assembly=True)
        if pair['id']==plan['reused_pair']:assert abs(scores['TOTAL']-plan['prior_per_head_reference'])<1e-9
        (STUDY/(pair['id']+'_RESULT.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');results.append(result)
    ranked=sorted(results,key=lambda r:r['scores']['TOTAL'],reverse=True)
    result=dict(scores=ranked[0]['scores'],best_pair=ranked[0]['id'],ranking=ranked,models=plan['models'],pairs=len(results),
                grid='0:0.01:1',selection='Per-head official whole-head score, fixed ratio for every row; exact ties prefer higher A weight',
                method='Native predictive probability mixture, unchanged per-head official utility argmax; utility weighting is linear-equivalent',
                limitation='Same fifth-fold labels used to choose pair and ratios; diagnostic only, no independent generalization/full OOF claim.',
                CPU_seconds=time.perf_counter()-started,GPU_pass_reports=gpu_reports,
                GPU_process_seconds_sum=sum(r['seconds'] for r in gpu_reports),
                study_wall_seconds=time.time()-plan['started_epoch'],timing_scope='Concurrent with ongoing serial training; includes implementation/preparation/waiting and evaluation, not solo benchmark',
                extra_training_stages=0,preexisting_training_stages_per_pair=16,screening_fold=5,
                complete_five_folds=False,remaining_folds_started=False,automatic_adoption=False,
                strictly_above_8074=ranked[0]['scores']['TOTAL']>80.74)
    (STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (STUDY/'WEIGHT_SCORES.csv').open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=('pair','target','weight_A','score'));writer.writeheader();writer.writerows(allrows)
    table='| A + B | 0.5/0.5 TOTAL | Best common TOTAL | Per-head TOTAL | POR A | PERM A | SW A |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in ranked:
        table+=f"| {r['A']} + {r['B']} | {r['equal_weight_scores']['TOTAL']:.9f} | {r['common_weight']['scores']['TOTAL']:.9f} | {r['scores']['TOTAL']:.12f} | "+' | '.join(f"{r['selected_weights'][t]['weight_A']:.2f}" for t in targets)+' |\n'
    note=f"\nCompleted {len(results)}-pair native distribution diagnostic over {len(plan['models'])} completed models / 144665 original fifth-fold rows. Best {ranked[0]['id']} TOTAL={ranked[0]['scores']['TOTAL']:.12f}; delta80.74={ranked[0]['delta_vs_8074']:+.12f}; delta prior80.735166={ranked[0]['delta_vs_prior_per_head']:+.12f}. All new native model predictions and grid endpoints exact, each row covered once; first pair reuses verified cache. Fixed whole-head ratios, grid0.01; no row/CONT-specific selection. No new training stages, 16 preexisting stages per pair, no other folds/adoption/fullOOF claim. Training continues. Timing concurrent GPU sum {result['GPU_process_seconds_sum']:.3f}s, CPU{result['CPU_seconds']:.3f}s, studywall {result['study_wall_seconds']:.3f}s including implementation/wait; no solo throughput claim.\n"
    (STUDY/'RESULTS.md').write_text(table+note,encoding='utf-8')
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as h:h.write(note)
    with (STUDY/'README.md').open('a',encoding='utf-8') as h:
        h.write('\n'+table+'\nNegative scope: lower-scoring pairs under this 0.01 native-distribution grid do not justify finer grids or new fusion mechanisms. Independent generalization and fullOOF remain unverified.\n')
    print(json.dumps(dict(best_pair=result['best_pair'],scores=result['scores'],ranking=[dict(pair=r['id'],TOTAL=r['scores']['TOTAL'],weights=r['selected_weights']) for r in ranked]),indent=2))

if __name__=='__main__':main()

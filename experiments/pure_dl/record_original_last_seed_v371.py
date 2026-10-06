"""CPU-only three-head assembly for the newly authorized original/seed14 pair."""
import time
from pathlib import Path
import numpy as np
from fullcv_pair_context_v370 import ROOT,read,write,sha
from train_invariant import metric,TS

study=ROOT/'results/deep_learning/original_last_seed_fusion_v371_20261002'
start=time.perf_counter();truth=np.load(ROOT/'results/deep_learning/pairwise_complete_models_v368_20261002/truth.npy')
assert truth.shape==(144665,3)
grids=[];native=[np.empty_like(truth),np.empty_like(truth)];curves={};selected={}
for c,target in enumerate(TS):
    report=read(study/(target+'_GPU_FINAL.json'))
    assert report['native_disk_replay_exact'] and report['endpoints_exact'] and report['coverage_exactly_once']
    with np.load(study/(target+'_grids.npz')) as z:
        pred=z['actions'][z['indices'][0]];weights=z['weights'];solo=z['native']
    assert pred.shape==(144665,101)
    scores=[]
    for i,w in enumerate(weights):
        candidate=truth.copy();candidate[:,c]=pred[:,i];scores.append(metric(truth,candidate)[target])
    best=max(range(101),key=lambda i:(scores[i],float(weights[i])))
    selected[target]=dict(weight_original=float(weights[best]),weight_seed14=float(1-weights[best]),score=scores[best],index=best)
    curves[target]=[dict(weight_original=float(w),score=float(s)) for w,s in zip(weights,scores)]
    for j in range(2):native[j][:,c]=solo[:,j]
    grids.append(pred)
common_scores=[]
for i in range(101):common_scores.append(metric(truth,np.column_stack([g[:,i] for g in grids])))
common=max(range(101),key=lambda i:(common_scores[i]['TOTAL'],float(weights[i])))
equal=np.column_stack([g[:,50] for g in grids]);joint=np.column_stack([g[:,common] for g in grids])
per_head=np.column_stack([g[:,selected[t]['index']] for t,g in zip(TS,grids)])
for label,pred in [('equal',equal),('common',joint),('per_head',per_head)]:
    np.save(study/(label+'_prediction.npy'),pred);assert np.array_equal(np.load(study/(label+'_prediction.npy')),pred)
scores=metric(truth,per_head);original=metric(truth,native[0]);new=metric(truth,native[1])
assert abs(original['TOTAL']-80.70612972323806)<1e-9
assert abs(new['TOTAL']-read(ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261014/COMPLETE_MODEL.json')['scores']['TOTAL'])<1e-9
prior=80.86345755738155
result=dict(pair='original__seed_20261014',screening_fold=5,rows=144665,standalone_original=original,
            standalone_seed14=new,equal_scores=metric(truth,equal),common_best_weight_original=float(weights[common]),
            common_scores=common_scores[common],per_head_scores=scores,per_head_weights=selected,
            delta_vs_original=scores['TOTAL']-original['TOTAL'],delta_vs_original_seed13= scores['TOTAL']-prior,
            native_disk_replay_exact=True,coverage_once=True,assembled_prediction_disk_readback_exact=True,
            scope='same fifth pair/weight tuning diagnostic, not fullCV',remaining_fold_pair_unchanged='original__seed_20261013',
            timing=dict(CPU_seconds=time.perf_counter()-start,GPU_phase_sum_seconds=sum(read(study/(t+'_GPU_FINAL.json'))['seconds'] for t in TS),
                        concurrency='three GPU heads serial; CPU preparation may overlap',includes='load/warmup/native forward/grid/write and CPU scoring'),
            metadata_correction='Inherited concurrent_training_continues=true in GPU reports is inapplicable: V362 was already complete for all three; no training active during scans.')
write(study/'FINAL.json',result);write(study/'WEIGHT_CURVES.json',curves)
table='|Method|POR|PERM|SW|TOTAL|\n|---|---:|---:|---:|---:|\n'
for name,s in [('Original',original),('Seed14',new),('Equal 50:50',result['equal_scores']),('Common best',result['common_scores']),('Per-head best',scores)]:
    table+='|'+name+'|'+ '|'.join(f'{s[t]:.6f}' for t in (*TS,'TOTAL'))+'|\n'
(study/'RESULTS.md').write_text(table+'\nWeights (original/seed14): '+str(selected)+'\n\nSame fifth tuning diagnostic. FullCV remains original+seed13.\n',encoding='utf-8')
note=f"\nV371 final per-head diagnostic TOTAL{scores['TOTAL']:.12f}; delta-original{result['delta_vs_original']:+.12f}; delta-original+seed13{result['delta_vs_original_seed13']:+.12f}. Native disk replay, endpoints, coverage and actual three-head assembly/readback exact. No extra training.\n"
for p in (study/'README.md',study/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(note)
print(result)

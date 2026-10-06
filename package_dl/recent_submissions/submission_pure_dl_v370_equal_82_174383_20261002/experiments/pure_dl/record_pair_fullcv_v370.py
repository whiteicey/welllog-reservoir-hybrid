"""CPU reports: fixed weights and outer-fold tuned diagnostic kept separate."""
import argparse
from pathlib import Path
import numpy as np
from fullcv_pair_context_v370 import ROOT,STUDY,OUTPUT,STAGES,BASE_SEEDS,read,write,sha,own_dir
from fullcv_context_v97 import original_population
from train_invariant import metric,TS

def record_model(identity,fold):
    own=own_dir(identity,fold)
    reports={s:read(own/s/(f'fold_{fold}_report.json' if s=='SOURCE' else 'report.json')) for s in STAGES}
    population=original_population();val=np.flatnonzero(population['fold']==fold-1)
    prediction=np.column_stack([np.load(own/s/'channel_pred.npy') for s in ('POR341','PERM336','SW351')])
    assert prediction.shape==(len(val),3) and np.isfinite(prediction).all()
    scores=metric(population['truth'][val],prediction)
    for c,s in enumerate(('POR341','PERM336','SW351')):
        assert reports[s]['predictions_reproduced_exactly']
        assert abs(reports[s]['target_score']-scores[TS[c]])<1e-9
    np.save(own/'complete_model_pred.npy',prediction)
    write(own/'COMPLETE_MODEL.json',dict(identity=identity,fold=fold,seed=BASE_SEEDS[identity]+fold-1,
          scores=scores,rows=len(val),stages=list(STAGES),training_stages=8,source_retrained=True,
          checkpoint_sha256={s:sha(own/s/(f'fold_{fold}.pt' if s=='SOURCE' else 'model_best.pt')) for s in STAGES},
          reports=reports,independent_disk_reload=False))

def record_fold(fold):
    plan=read(STUDY/'PLAN.json');pop=original_population();val=np.flatnonzero(pop['fold']==fold-1);truth=pop['truth'][val]
    dest=STUDY/f'fold_{fold}';dest.mkdir(exist_ok=True)
    fixed=np.empty_like(truth);tuned=np.empty_like(truth);native=[np.empty_like(truth),np.empty_like(truth)]
    selected={};curves={}
    for c,target in enumerate(TS):
        if fold==5:
            parent=ROOT/'results/deep_learning/pairwise_complete_models_v368_20261002'
            with np.load(parent/('EXTRA_'+target+'_grids.npz')) as z:
                actions=z['actions'];indices=z['indices'][0];weights=z['weights'];solo=z['native'][:,[0,4]]
            augmented=read(parent/'AUGMENT_PLAN.json')
            assert augmented['pairs'][0]['id']=='original__seed_20261013'
            assert [s['id'] for s in augmented['models']][::4]==['original','seed_20261013']
            evidence=read(parent/('EXTRA_'+target+'_GPU_FINAL.json'))
            assert evidence['native_disk_replay_exact'] and evidence['endpoints_exact']
        else:
            with np.load(dest/(target+'_grids.npz')) as z:
                actions=z['actions'];indices=z['indices'];weights=z['weights'];solo=z['native']
                assert np.array_equal(z['validation'],val)
            evidence=read(dest/(target+'_GPU_REPORT.json'))
            assert evidence['native_disk_replay_exact'] and evidence['coverage_once']
        assert indices.shape==(len(val),101) and solo.shape==(len(val),2)
        assert np.array_equal(actions[indices[:,0]],solo[:,1])
        assert np.array_equal(actions[indices[:,-1]],solo[:,0])
        scores=[]
        for i,w in enumerate(weights):
            candidate=truth.copy();candidate[:,c]=actions[indices[:,i]]
            scores.append(float(metric(truth,candidate)[target]))
        # Deterministic equality ties prefer greater original weight.
        best=max(range(101),key=lambda i:(scores[i],float(weights[i])))
        fixed_index=int(round(plan['fixed_weight_A'][target]*100))
        assert abs(float(weights[fixed_index])-plan['fixed_weight_A'][target])<1e-12
        fixed[:,c]=actions[indices[:,fixed_index]];tuned[:,c]=actions[indices[:,best]]
        for j in range(2):native[j][:,c]=solo[:,j]
        curves[target]=[dict(weight_original=float(w),weight_seed13=float(1-w),score=s) for w,s in zip(weights,scores)]
        selected[target]=dict(weight_original=float(weights[best]),weight_seed13=float(1-weights[best]),
                              score=scores[best],fixed_weight_score=scores[fixed_index])
    for label,pred in [('fixed',fixed),('diagnostic',tuned),('original',native[0]),('seed13',native[1])]:
        p=dest/(label+'_prediction.npy');np.save(p,pred);assert np.array_equal(np.load(p),pred)
    if fold==5:
        assert abs(metric(truth,tuned)['TOTAL']-80.86345755738155)<1e-9
    result=dict(fold=fold,rows=len(val),fixed_weights=plan['fixed_weight_A'],fixed_scores=metric(truth,fixed),
          diagnostic_scores=metric(truth,tuned),per_head_best_weights=selected,
          standalone_scores={'original':metric(truth,native[0]),'seed13':metric(truth,native[1])},
          native_replay_exact=True,prediction_disk_readback_exact=True,
          diagnostic_scope='This fold labels selected each head weight; tuning diagnostic, not independent validation',
          fixed_scope='Fifth-selected fixed ratios applied unchanged; fifth itself remains selection-contaminated')
    write(dest/'FOLD_REPORT.json',result);write(dest/'WEIGHT_CURVES.json',curves)
    note=f"\nV370 fold{fold} complete, {len(val)} rows. Fixed TOTAL{result['fixed_scores']['TOTAL']:.12f}; foldwise tuned diagnostic TOTAL{result['diagnostic_scores']['TOTAL']:.12f}; weights{selected}. Exact native disk replay and output readback; no rowwise/stratum choice.\n"
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as h:h.write(note)
    return result

def record_oof():
    pop=original_population();n=len(pop['truth']);coverage=np.zeros(n,np.int8)
    preds={k:np.full_like(pop['truth'],np.nan) for k in ('fixed','diagnostic','original','seed13')}
    reports=[]
    for f in range(1,6):
        dest=STUDY/f'fold_{f}';report=read(dest/'FOLD_REPORT.json');reports.append(report)
        ix=np.flatnonzero(pop['fold']==f-1);assert len(ix)==report['rows'] and (coverage[ix]==0).all();coverage[ix]+=1
        for key in preds:preds[key][ix]=np.load(dest/(key+'_prediction.npy'))
    assert (coverage==1).all() and n==723556
    scores={k:metric(pop['truth'],v) for k,v in preds.items()}
    for k,v in preds.items():
        assert np.isfinite(v).all();np.save(STUDY/(k+'_OOF.npy'),v)
        assert np.array_equal(np.load(STUDY/(k+'_OOF.npy')),v)
    np.savez_compressed(STUDY/'OOF_POPULATION.npz',truth=pop['truth'],fold=pop['fold'],wells=pop['wells'],valid_to_raw=pop['valid_to_raw'])
    result=dict(scores=scores,folds=reports,rows=n,complete_original_five_fold_OOF=True,coverage_once=True,
          disk_readback_exact=True,fixed_strictly_above82=scores['fixed']['TOTAL']>82,
          diagnostic_above82=scores['diagnostic']['TOTAL']>82,
          caution='Pair and fixed ratios selected on fifth; tuned OOF additionally selects weights on every scored fold. Diagnostic max is not an unbiased >82 achievement.',
          original_split=True,test_labels_accessed=False,stages_per_model=8,historical_stages_per_pair=16)
    write(STUDY/'FINAL.json',result)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=range(1,6));ap.add_argument('--identity',choices=BASE_SEEDS);ap.add_argument('--final',action='store_true');a=ap.parse_args()
    if a.final:print(record_oof())
    elif a.identity:record_model(a.identity,a.fold)
    else:print(record_fold(a.fold))

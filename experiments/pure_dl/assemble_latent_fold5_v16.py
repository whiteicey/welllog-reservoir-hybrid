"""Reload every candidate SW and the fixed private POR/PERM, then score real arrays.

No fitting or validation-dependent row routing. A whole target model is selected
only after complete evaluation. Old submitted artifacts are never overwritten.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import gc, json, time
import numpy as np
import torch
import psutil
from train_latent_fold5_v16 import ROOT, STUDY, REFERENCE, MODES, make_model, load_outer_data
from train_sw_group_v14 import Evaluator as GridEvaluator
from train_independent_channels_v8 import Evaluator as OriginalEvaluator, REFERENCE as HEAD_REFERENCE
from train_dual8222_heads80 import SOURCE, prepare_data, atomic_json, atomic_array
from independent_channels_v8 import Network as PrivateModel
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def main():
    for mode in MODES:
        out=ROOT/'outputs'/('pure_dl_latent_v16_'+mode+'_fold5')
        if not (out/'fold_5_report.json').exists(): raise RuntimeError('Unfinished candidate: '+mode)
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=proc.info;argv=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(d['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    if active:raise RuntimeError('Do not compete with active training during final verification: '+str(active))
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    data=load_outer_data(); evaluator=GridEvaluator(data,data['monitor_raw'])
    ref=np.load(REFERENCE/'fold_5_pred.npy')
    with np.load(REFERENCE/'oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==(144665,3) and ref.shape==truth.shape
    assert np.array_equal(data['truth'][data['monitor_raw']],truth[:,2])
    assert np.array_equal(data['wells'][data['monitor_raw']],wells)
    models={};predictions={};timings=[];reports={}
    for mode in MODES:
        out=ROOT/'outputs'/('pure_dl_latent_v16_'+mode+'_fold5')
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model=make_model(mode,cp['centers'],cp['target_median'],cp['target_scale']).cuda()
        model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        tick=time.perf_counter();pred,stats=evaluator(model)
        assert np.array_equal(pred,np.load(out/'channel_pred.npy'))
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        assert abs(stats['SW']-report['target_score'])<1e-9
        models['SW_'+mode]=model;predictions[mode]=pred;reports[mode]=report
        timings.append(dict(target='SW',mode=mode,replay_seconds=time.perf_counter()-tick,all144665rows_exact=True))
        print('SW_RELOADED',mode,stats['SW'],flush=True)
        del cp
    del data,evaluator;gc.collect();torch.cuda.empty_cache()
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    head=torch.load(HEAD_REFERENCE/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
    legacy=prepare_data(master)
    assert np.array_equal(legacy['truth'][legacy['validation']],truth)
    assert np.array_equal(legacy['wells'][legacy['validation']],wells)
    manifest=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))
    fixed=np.zeros((144665,2),np.float64)
    for c,name in enumerate(('POR','PERM')):
        choice=next(x for x in manifest['choices'] if x['target']==name)
        cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False)
        model=PrivateModel(master,head,name,choice['architecture']).cuda()
        model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        ev=OriginalEvaluator(legacy,master,c,ref)
        tick=time.perf_counter();pred,score,total=ev(model)
        assert np.array_equal(pred,ref[:,c])
        models[name]=model;fixed[:,c]=pred
        timings.append(dict(target=name,replay_seconds=time.perf_counter()-tick,all144665rows_exact=True))
        print('FIXED_MODEL_RELOADED',name,score,flush=True)
        del cp,ev
    refs=base.metric(truth,ref)
    candidates=[]
    for mode in MODES:
        relevant=[models['POR'],models['PERM'],models['SW_'+mode]]
        storage=[{p.untyped_storage().data_ptr() for p in model.parameters() if p.numel()>0} for model in relevant]
        assert not any(storage[i]&storage[j] for i in range(3) for j in range(i))
        pred=np.column_stack([fixed,predictions[mode]])
        scores=base.metric(truth,pred)
        assert abs(scores['TOTAL']-reports[mode]['reference_replacement_total'])<1e-8
        atomic_array(STUDY/('assembled_'+mode+'_pred.npy'),pred)
        candidate=dict(mode=mode,scores=scores,delta_v5_TOTAL=scores['TOTAL']-refs['TOTAL'],
            delta_old_SW=scores['SW']-refs['SW'],three_models_loaded_from_disk=True,
            no_shared_parameter_storage=True,POR_PERM_exactly_unchanged=True,all144665SW_predictions_exact=True,
            private_backbone_branches=5,checkpoint='outputs/pure_dl_latent_v16_'+mode+'_fold5/model_best.pt')
        candidates.append(candidate)
    best=max(candidates,key=lambda x:x['scores']['TOTAL'])
    adopted=best['scores']['TOTAL']>refs['TOTAL']
    chosen=[x.copy() for x in manifest['choices']]
    if adopted:
        for choice in chosen:
            if choice['target']=='SW':
                choice.update(key='SW_v16_'+best['mode'],architecture=best['mode'],checkpoint=best['checkpoint'],
                    model_module='train_latent_fold5_v16',builder='make_model',target_score=best['scores']['SW'],
                    best_epoch=reports[best['mode']]['best_epoch'])
    final_scores=best['scores'] if adopted else refs
    recipe=dict(scores=final_scores,choices=chosen,selected_new_SW=adopted,
        best_new_candidate=best['mode'],preprocessing_metadata='results/deep_learning/latent_fold5_v16_20260928/data_outer_manifest.json' if adopted else None,
        remaining_folds_started=False,complete_five_folds=False,development_selected_fold5=True,
        old_delivery_unchanged=True,current_model_policy='Keep v5/v8 when no completed new candidate exceeds it')
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    report=dict(fold=5,rows=144665,reference_scores=refs,candidates=candidates,
        selected_scores=final_scores,selected_new_SW=adopted,reference_best_v5_total=refs['TOTAL'],
        all_original_row_order_verified=True,all_new_models_and_fixed_POR_PERM_reloaded=True,
        timing=timings,total_setup_and_replay_seconds=time.perf_counter()-started,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(STUDY/'assembled_report.json',report)
    print('ASSEMBLED',json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__': main()

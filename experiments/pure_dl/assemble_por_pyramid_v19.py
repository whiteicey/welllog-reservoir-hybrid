"""Reload new v19 POR checkpoints and score real three-column arrays.

Unchanged PERM/SW arrays reuse their recorded disk-replay provenance. Only the
new POR models are freshly reloaded here; no fitting or rowwise model choices.
"""
from datetime import datetime
import json,sys,time
import numpy as np
import torch
from por_pyramid_v19 import Network,MODES,downsample
from por_data_v18 import ROOT,REFERENCE,load_data,Evaluator
from report_por_pyramid_v19 import STUDY
from train_dual8222_heads80 import atomic_json,atomic_array
from multiscale_sequence import window_indices
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
from assemble_por_architecture_v18 import check_model,points
from architecture_zoo_v15 import clean
from torch.nn import functional as F
import train_invariant as base


@torch.no_grad()
def context_probe(model,data):
    anchors=data['fit_raw'][:2]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    x=data['x'][ix];context=model.blocks['context'];model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        full=model(x,rel,valid)
        context.disabled_for_diagnostic=True
        try:disabled=model(x,rel,valid)
        finally:context.disabled_for_diagnostic=False
        h=clean(F.gelu(model.stem(clean(torch.cat((x.float(),rel[...,None]),-1),valid))),valid)
        for block in model.blocks['encoder0']:h=block(h,valid)
        h,m1=downsample(h,valid)
        for block in model.blocks['encoder1']:h=block(h,m1)
        h,m2=downsample(h,m1)
        for block in model.blocks['encoder2']:h=block(h,m2)
        gates=context.gate(clean(context.norm(h),m2)).sigmoid()[m2].float()
    masks=context.masks(64,x.device)
    counts=[int(m.sum()) for m in masks]
    if model.mode=='global':assert counts==[1024,256,1024]
    else:assert all(0<n<cap for n,cap in zip(counts,[1024,256,1024]))
    return dict(training_windows=2,gate_mean=float(gates.mean()),gate_min=float(gates.min()),gate_max=float(gates.max()),
        allowed_read_process_write_pairs=counts,
        disabled_gate_logit_mean_change=float((full[0]-disabled[0])[valid].abs().mean()),
        disabled_distribution_raw_mean_change=float((full[1]-disabled[1])[valid].abs().mean()),
        interpretation='Small train-input forward dependency probe; no retraining or scored extra candidate')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Final replay does not compete with active trainers')
    for m in MODES:
        if not (ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{m}_fold5/fold_5_report.json').exists():raise RuntimeError('Unfinished '+m)
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter();data=load_data()
    reference=np.load(REFERENCE/'fold_5_pred.npy')
    with np.load(REFERENCE/'oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    ids=data['monitor_raw'];default=data['default_truth'][ids]
    assert truth.shape==reference.shape==(144665,3)
    assert np.array_equal(truth[:,0],data['truth'][ids]) and np.array_equal(wells,data['wells'][ids])
    ref_scores=base.metric(truth,reference);old=points(truth[:,0],reference[:,0]);predictions={};candidates=[]
    for m in MODES:
        out=ROOT/'outputs'/f'pure_dl_por_pyramid_v19_{m}_fold5'
        checkpoint=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model=Network(m,checkpoint['metadata']).cuda();model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        ev=Evaluator(data,ids,checkpoint['metadata']);tick=time.perf_counter()
        prediction,metrics=ev(model)
        assert np.array_equal(prediction,np.load(out/'channel_pred.npy'))
        checks=check_model(model,data);checks['context']=context_probe(model,data)
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        assert abs(metrics['POR']-report['target_score'])<1e-9
        combined=reference.copy();combined[:,0]=prediction
        assert np.array_equal(combined[:,1:],reference[:,1:])
        scores=base.metric(truth,combined)
        assert abs(scores['TOTAL']-report['reference_replacement_total'])<1e-8
        current=points(truth[:,0],prediction);delta=current-old
        pw=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
        atomic_array(STUDY/f'assembled_{m}_pred.npy',combined)
        c=dict(mode=m,scores=scores,delta_v5_TOTAL=scores['TOTAL']-ref_scores['TOTAL'],delta_old_POR=scores['POR']-ref_scores['POR'],
            best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],trainable_parameters=report['trainable_parameters'],
            new_POR_reloaded_from_disk=True,all144665POR_predictions_exact=True,fixed_PERM_SW_arrays_reused=True,
            fixed_PERM_SW_fresh_GPU_replay=False,PERM_SW_exactly_unchanged=True,parameter_sharing_between_targets=False,
            verification=checks,replay_and_checks_seconds=time.perf_counter()-tick,
            populations=dict(default_points=int(default.sum()),numeric_points=int((~default).sum()),
                default_POR=float(current[default].mean()),numeric_POR=float(current[~default].mean()),
                default_delta=float(delta[default].mean()),numeric_delta=float(delta[~default].mean())),
            well_wins=sum(v['POR_delta']>0 for v in pw),well_losses=sum(v['POR_delta']<0 for v in pw),per_well=pw,
            checkpoint=f'outputs/pure_dl_por_pyramid_v19_{m}_fold5/model_best.pt')
        candidates.append(c);predictions[m]=prediction
        print('CANDIDATE',json.dumps({k:v for k,v in c.items() if k!='per_well'},ensure_ascii=False),flush=True)
        del model,ev,checkpoint;torch.cuda.empty_cache()
    delta=points(truth[:,0],predictions['global'])-points(truth[:,0],predictions['regional'])
    pw=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
    sizes=np.array([v['n'] for v in pw]);diff=np.array([v['POR_delta'] for v in pw])
    sample=np.random.default_rng(20261010).integers(0,16,size=(10000,16))
    boot=(diff[sample]*sizes[sample]).sum(1)/sizes[sample].sum(1)*.3
    comparison=dict(global_minus_regional_POR=float(delta.mean()),global_minus_regional_TOTAL=float(delta.mean()*.3),
        default_POR_delta=float(delta[default].mean()),numeric_POR_delta=float(delta[~default].mean()),
        default_TOTAL_contribution=float(delta[default].sum()/len(delta)*.3),numeric_TOTAL_contribution=float(delta[~default].sum()/len(delta)*.3),
        well_wins=sum(v['POR_delta']>0 for v in pw),well_losses=sum(v['POR_delta']<0 for v in pw),per_well=pw,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]).tolist())
    best=max(candidates,key=lambda c:c['scores']['TOTAL']);adopt=best['scores']['TOTAL']>ref_scores['TOTAL']+1e-9
    old_manifest=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'));choices=[dict(c) for c in old_manifest['choices']]
    if adopt:
        for c in choices:
            if c['target']=='POR':c.update(key='POR_v19_'+best['mode'],architecture=best['mode'],checkpoint=best['checkpoint'],
                model_module='por_pyramid_v19',builder='Network(mode,metadata)',target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected=best['scores'] if adopt else ref_scores
    previous=json.loads((ROOT/'results/deep_learning/por_architecture_v18_20260928/assembled_report.json').read_text(encoding='utf-8'))
    prior_comparison=[dict(mode=c['mode'],TOTAL_delta_vs_v18_tcn=c['scores']['TOTAL']-previous['candidates'][0]['scores']['TOTAL'],
        note='Architecture and parameter count both differ; not matched module attribution') for c in candidates]
    result=dict(fold=5,rows=144665,reference_scores=ref_scores,candidates=candidates,comparison=comparison,versus_v18=prior_comparison,
        selected_scores=selected,selected_new_POR=adopt,all_new_POR_reloaded=True,fixed_other_predictions_reused=True,
        fixed_other_replay_evidence=['results/deep_learning/latent_gate_v17_20260928/assembled_report.json',
            'results/deep_learning/latent_fold5_v16_20260928/assembled_report.json'],
        row_order_verified=True,seconds=time.perf_counter()-started,finished_at=datetime.now().astimezone().isoformat(),
        complete_five_folds=False,remaining_folds_started=False)
    recipe=dict(scores=selected,choices=choices,selected_new_POR=adopt,best_new_candidate=best['mode'],
        metadata='results/deep_learning/por_architecture_v18_20260928/model_metadata.pt' if adopt else None,
        feature_metadata='results/deep_learning/latent_fold5_v16_20260928/data_outer_manifest.json' if adopt else None,
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('SELECTED',json.dumps(recipe,ensure_ascii=False));print('COMPARISON',json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':main()

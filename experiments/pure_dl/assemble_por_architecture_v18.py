"""Replay new POR models and combine with already-verified fixed PERM/SW arrays.

Does not refit, pick per-row models or claim fresh old-model GPU replay.
"""
from __future__ import annotations
from datetime import datetime
import json,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from por_architecture_v18 import Network,MODES
from por_data_v18 import ROOT,STUDY,REFERENCE,load_data,Evaluator
from train_dual8222_heads80 import atomic_json,atomic_array
from multiscale_sequence import window_indices
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def points(y,p):
    return 100*np.maximum(0,1-np.abs(p-y)/(.08*(np.abs(y)+1e-5)))


@torch.no_grad()
def check_model(model,d):
    anchors=d['fit_raw'][:2]
    ix,valid=window_indices(anchors,d['lower'],d['upper'])
    rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
    x=d['x'][ix]
    model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        normal=model(x,rel,valid)
        changed=model(x.float().masked_fill(~valid[...,None],10000),rel.masked_fill(~valid,-10000),valid)
        empty=model(x,rel,torch.zeros_like(valid))
    error=max(float((a-b).abs().max()) for a,b in zip(normal,changed))
    assert error==0 and all(bool(torch.isfinite(a).all()) and bool((a==0).all()) for a in empty)
    r=normal[1][valid][:16].float();z=d['labels'][ix][valid][:16].float()
    lw,mu,sd=model.distribution_parameters(r)
    expected=-torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=lw),
        torch.distributions.StudentT(3.,mu,sd/(3**.5))).log_prob(z)
    nll_error=float((expected-model.log_components(r,z)).abs().max())
    assert nll_error<1e-5
    return dict(training_input_windows=2,padding_max_error=error,empty_outputs_finite_zero=True,
                NLL_vs_torch_StudentT_max_error=nll_error)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Wait until existing project trainers have finished')
    for m in MODES:
        if not (ROOT/'outputs'/f'pure_dl_por_architecture_v18_{m}_fold5/fold_5_report.json').exists():raise RuntimeError(f'Unfinished {m}')
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter();d=load_data()
    reference_pred=np.load(REFERENCE/'fold_5_pred.npy')
    with np.load(REFERENCE/'oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    ids=d['monitor_raw'];default=d['default_truth'][ids]
    assert reference_pred.shape==truth.shape==(144665,3)
    assert np.array_equal(truth[:,0],d['truth'][ids]) and np.array_equal(wells,d['wells'][ids])
    refscore=base.metric(truth,reference_pred)
    old=points(truth[:,0],reference_pred[:,0]);predictions={};candidates=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_por_architecture_v18_{mode}_fold5'
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model=Network(mode,cp['metadata']).cuda();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        evaluator=Evaluator(d,ids,cp['metadata']);tick=time.perf_counter()
        pred,stats=evaluator(model)
        assert np.array_equal(pred,np.load(out/'channel_pred.npy'))
        checks=check_model(model,d)
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        assert abs(stats['POR']-report['target_score'])<1e-9
        combined=reference_pred.copy();combined[:,0]=pred
        assert np.array_equal(combined[:,1:],reference_pred[:,1:])
        scores=base.metric(truth,combined)
        assert abs(scores['TOTAL']-report['reference_replacement_total'])<1e-8
        new=points(truth[:,0],pred);delta=new-old
        perwell=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
        atomic_array(STUDY/f'assembled_{mode}_pred.npy',combined)
        candidate=dict(mode=mode,scores=scores,delta_v5_TOTAL=scores['TOTAL']-refscore['TOTAL'],
            delta_old_POR=scores['POR']-refscore['POR'],best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],
            trainable_parameters=report['trainable_parameters'],new_POR_reloaded_from_disk=True,all144665POR_predictions_exact=True,
            fixed_PERM_SW_arrays_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,PERM_SW_exactly_unchanged=True,
            parameter_sharing_between_targets=False,verification=checks,replay_and_checks_seconds=time.perf_counter()-tick,
            populations=dict(default_points=int(default.sum()),numeric_points=int((~default).sum()),
                default_POR=float(new[default].mean()),numeric_POR=float(new[~default].mean()),
                default_delta=float(delta[default].mean()),numeric_delta=float(delta[~default].mean()),
                default_TOTAL_contribution=float(delta[default].sum()/len(delta)*.3),numeric_TOTAL_contribution=float(delta[~default].sum()/len(delta)*.3)),
            well_wins=sum(x['POR_delta']>0 for x in perwell),well_losses=sum(x['POR_delta']<0 for x in perwell),per_well=perwell,
            checkpoint=f'outputs/pure_dl_por_architecture_v18_{mode}_fold5/model_best.pt')
        predictions[mode]=pred;candidates.append(candidate)
        print('CANDIDATE',json.dumps({k:v for k,v in candidate.items() if k!='per_well'},ensure_ascii=False),flush=True)
        del model,evaluator,cp
        torch.cuda.empty_cache()
    diff=points(truth[:,0],predictions['latent64'])-points(truth[:,0],predictions['tcn104'])
    perwell=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(diff[wells==w].mean())) for w in np.unique(wells)]
    rng=np.random.default_rng(20261010);samples=rng.integers(0,16,size=(10000,16))
    n=np.array([x['n'] for x in perwell]);ds=np.array([x['POR_delta'] for x in perwell])
    boot=(ds[samples]*n[samples]).sum(1)/n[samples].sum(1)*.3
    comparison=dict(latent_minus_tcn_POR=float(diff.mean()),latent_minus_tcn_TOTAL=float(diff.mean()*.3),
        default_POR_delta=float(diff[default].mean()),numeric_POR_delta=float(diff[~default].mean()),
        well_wins=sum(x['POR_delta']>0 for x in perwell),well_losses=sum(x['POR_delta']<0 for x in perwell),per_well=perwell,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]).tolist())
    best=max(candidates,key=lambda x:x['scores']['TOTAL']);adopt=best['scores']['TOTAL']>refscore['TOTAL']+1e-9
    manifest=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))
    choices=[dict(x) for x in manifest['choices']]
    if adopt:
        for c in choices:
            if c['target']=='POR':c.update(key='POR_v18_'+best['mode'],architecture=best['mode'],checkpoint=best['checkpoint'],
                model_module='por_architecture_v18',builder='Network(mode,metadata)',target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected=best['scores'] if adopt else refscore
    result=dict(fold=5,rows=len(truth),reference_scores=refscore,candidates=candidates,comparison=comparison,
        selected_scores=selected,selected_new_POR=adopt,all_new_POR_reloaded=True,fixed_other_predictions_reused=True,
        fixed_other_replay_evidence='results/deep_learning/latent_gate_v17_20260928/assembled_report.json',
        row_order_verified=True,seconds=time.perf_counter()-started,finished_at=datetime.now().astimezone().isoformat(),
        complete_five_folds=False,remaining_folds_started=False)
    recipe=dict(scores=selected,choices=choices,selected_new_POR=adopt,best_new_candidate=best['mode'],
        metadata='results/deep_learning/por_architecture_v18_20260928/model_metadata.pt' if adopt else None,
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('SELECTED',json.dumps(recipe,ensure_ascii=False))
    print('COMPARISON',json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':main()

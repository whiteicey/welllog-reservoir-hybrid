"""Reload new POR models, score fixed three-column predictions, separate two effects."""
from datetime import datetime
import json,sys,time,gc
import numpy as np
import torch
from feature_structure_v20 import Network,MODES,STUDY,load_feature_data
from por_data_v18 import ROOT,REFERENCE,Evaluator
from train_dual8222_heads80 import atomic_json,atomic_array
from multiscale_sequence import window_indices
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
from assemble_por_architecture_v18 import check_model,points
import train_invariant as base

@torch.no_grad()
def check_views(model,d):
    anchors=d['fit_raw'][:2];ix,valid=window_indices(anchors,d['lower'],d['upper'])
    x=d['x'][ix];rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
    model.eval()
    perturbed=x.clone();perturbed[...,206:]=0
    with torch.autocast('cuda',dtype=torch.bfloat16):
        a,b,g=model.encoded_views(x,rel,valid)
        aa,bb,gg=model.encoded_views(perturbed,rel,valid)
        output=model(x,rel,valid);other=model(perturbed,rel,valid)
    main_delta=float((a-aa)[valid].abs().max())
    if model.mode=='shape_late':assert main_delta==0
    return dict(training_windows=2,main_encoder_side_input_perturbation_max_change=main_delta,
        side_encoder_perturbation_mean_change=float((b-bb)[valid].abs().mean()),
        fusion_gate_mean=float(g[valid].float().mean()),fusion_gate_min=float(g[valid].float().min()),fusion_gate_max=float(g[valid].float().max()),
        gate_output_change=float((output[0]-other[0])[valid].abs().mean()),
        density_raw_change=float((output[1]-other[1])[valid].abs().mean()),
        scope='Training-input dependency probe only; not a retrained ablation or an extra scoring candidate')

def paired(left,right,truth,default,wells):
    delta=points(truth,right)-points(truth,left)
    pw=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
    n=np.array([r['n'] for r in pw]);ds=np.array([r['POR_delta'] for r in pw])
    sample=np.random.default_rng(20261010).integers(0,len(pw),size=(10000,len(pw)))
    boot=(ds[sample]*n[sample]).sum(1)/n[sample].sum(1)*.3
    return dict(POR_delta=float(delta.mean()),TOTAL_delta=float(delta.mean()*.3),
        default_POR_delta=float(delta[default].mean()),numeric_POR_delta=float(delta[~default].mean()),
        default_TOTAL_contribution=float(delta[default].sum()/len(delta)*.3),numeric_TOTAL_contribution=float(delta[~default].sum()/len(delta)*.3),
        well_wins=sum(r['POR_delta']>0 for r in pw),well_losses=sum(r['POR_delta']<0 for r in pw),per_well=pw,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]).tolist())

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('No replay alongside active trainers')
    for m in MODES:
        if not (ROOT/'outputs'/f'pure_dl_feature_structure_v20_{m}_fold5/fold_5_report.json').exists():raise RuntimeError('Unfinished '+m)
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter();reference=np.load(REFERENCE/'fold_5_pred.npy')
    with np.load(REFERENCE/'oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    ref_scores=base.metric(truth,reference)
    predictions={};candidates=[];defaults=None
    for mode in MODES:
        d=load_feature_data(mode);ids=d['monitor_raw'];default=d['default_truth'][ids];defaults=default
        assert np.array_equal(truth[:,0],d['truth'][ids]) and np.array_equal(wells,d['wells'][ids])
        out=ROOT/'outputs'/f'pure_dl_feature_structure_v20_{mode}_fold5'
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model=Network(mode,cp['metadata']).cuda();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        ev=Evaluator(d,ids,cp['metadata']);tick=time.perf_counter();pred,stats=ev(model)
        assert np.array_equal(pred,np.load(out/'channel_pred.npy'))
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        assert abs(stats['POR']-report['target_score'])<1e-9
        checks=check_model(model,d);checks['views']=check_views(model,d)
        combined=reference.copy();combined[:,0]=pred
        assert np.array_equal(combined[:,1:],reference[:,1:])
        score=base.metric(truth,combined)
        assert abs(score['TOTAL']-report['reference_replacement_total'])<1e-8
        atomic_array(STUDY/f'assembled_{mode}_pred.npy',combined)
        candidate=dict(mode=mode,scores=score,delta_v5_TOTAL=score['TOTAL']-ref_scores['TOTAL'],
            delta_old_POR=score['POR']-ref_scores['POR'],best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],
            trainable_parameters=report['trainable_parameters'],new_POR_disk_reloaded=True,all144665POR_predictions_exact=True,
            fixed_PERM_SW_predictions_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,PERM_SW_exactly_unchanged=True,
            independent_target_parameters=True,verification=checks,replay_and_checks_seconds=time.perf_counter()-tick,
            default_POR=stats['default_POR'],numeric_POR=stats['numeric_POR'],
            against_reference=paired(reference[:,0],pred,truth[:,0],default,wells),
            checkpoint=f'outputs/pure_dl_feature_structure_v20_{mode}_fold5/model_best.pt')
        candidates.append(candidate);predictions[mode]=pred
        print('CANDIDATE',json.dumps({k:v for k,v in candidate.items() if k!='against_reference'},ensure_ascii=False),flush=True)
        del cp,model,ev,d;gc.collect();torch.cuda.empty_cache()
    comparison=dict(feature_shape_minus_amplitude=paired(predictions['amplitude_early'],predictions['shape_early'],truth[:,0],defaults,wells),
        fusion_late_minus_early=paired(predictions['shape_early'],predictions['shape_late'],truth[:,0],defaults,wells))
    best=max(candidates,key=lambda c:c['scores']['TOTAL']);adopt=best['scores']['TOTAL']>ref_scores['TOTAL']+1e-9
    old=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'));choices=[dict(c) for c in old['choices']]
    if adopt:
        for c in choices:
            if c['target']=='POR':c.update(key='POR_v20_'+best['mode'],architecture=best['mode'],checkpoint=best['checkpoint'],
                model_module='feature_structure_v20',builder='Network(mode,metadata)',target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected=best['scores'] if adopt else ref_scores
    report=dict(fold=5,rows=144665,reference_scores=ref_scores,candidates=candidates,comparisons=comparison,
        selected_scores=selected,selected_new_POR=adopt,all_new_POR_reloaded=True,original_row_order_verified=True,
        fixed_other_predictions_reused=True,fixed_other_replay_evidence=['results/deep_learning/latent_gate_v17_20260928/assembled_report.json','results/deep_learning/latent_fold5_v16_20260928/assembled_report.json'],
        seconds=time.perf_counter()-started,finished_at=datetime.now().astimezone().isoformat(),complete_five_folds=False,remaining_folds_started=False)
    recipe=dict(scores=selected,choices=choices,selected_new_POR=adopt,best_new_candidate=best['mode'],
        side_features_metadata='results/deep_learning/feature_structure_v20_20260928/feature_manifest.json' if adopt else None,
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',report);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('SELECTED',json.dumps(recipe,ensure_ascii=False))
    print('CONTRASTS',json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_well'} for k,v in comparison.items()},ensure_ascii=False))

if __name__=='__main__':main()

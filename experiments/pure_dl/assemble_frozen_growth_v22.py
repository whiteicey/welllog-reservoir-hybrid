"""Verify actual post-update v22 checkpoints, score selected and updated arrays.

Unchanged PERM/SW arrays retain their earlier verification provenance. Epoch0
fallback scores are separated from post-update performance and costs.
"""
from datetime import datetime
from pathlib import Path
import gc, hashlib, json, sys, time
import numpy as np
import torch
from frozen_growth_v22 import make_model, load_data
from champion_scaling_v21 import REFERENCE
from report_frozen_growth_v22 import ROOT, STUDY, MODES, read_json
from train_scaling_v21 import Evaluator
from train_dual8222_heads80 import atomic_json, atomic_array
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def point_scores(y, p):
    return 100*np.maximum(0, 1-np.abs(p-y)/(.08*(np.abs(y)+1e-5)))


def breakdown(y, old, new, defaults, wells):
    delta=point_scores(y,new)-point_scores(y,old)
    pw=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
    return dict(POR_delta=float(delta.mean()),TOTAL_delta=float(.3*delta.mean()),
        default_POR_delta=float(delta[defaults].mean()),numeric_POR_delta=float(delta[~defaults].mean()),
        default_TOTAL_contribution=float(.3*delta[defaults].sum()/len(delta)),
        numeric_TOTAL_contribution=float(.3*delta[~defaults].sum()/len(delta)),
        well_wins=sum(r['POR_delta']>0 for r in pw),well_losses=sum(r['POR_delta']<0 for r in pw),per_well=pw)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs(): raise RuntimeError('Do not run final replay concurrently with active training')
    for mode in MODES:
        if not (ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{mode}_fold5/fold_5_report.json').exists():
            raise RuntimeError('Incomplete candidate '+mode)
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter();data,master=load_data()
    reference=np.load(REFERENCE/'fold_5_pred.npy')
    truth=data['truth'][data['validation']];wells=data['wells'][data['validation']]
    assert truth.shape==reference.shape==(144665,3)
    defaults=base.dmask(truth)[:,0]
    evaluator=Evaluator(data,master,0,reference)
    refscore=base.metric(truth,reference)
    candidates=[];updated_preds={}
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_frozen_growth_v22_{mode}_fold5'
        report=read_json(out/'fold_5_report.json');cfg=read_json(out/'config.json')
        for name,signature in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==signature, name
        # Rebuild the trained, best-after-update candidate, even when the final selection is epoch0.
        model=make_model(mode,master).cuda()
        original={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if '.extra_' not in n}
        checkpoint=torch.load(out/'model_best_updated.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in original.items())
        tick=time.perf_counter();updated,score,total=evaluator(model)
        assert np.array_equal(updated,np.load(out/'best_updated_pred.npy'))
        assert abs(score-report['best_after_update']['target_score'])<1e-9
        replay_seconds=time.perf_counter()-tick
        selected_cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        assert selected_cp['best_epoch']==report['best_epoch']
        assert all(torch.equal(selected_cp['state_dict'][n].cpu(),v) for n,v in original.items())
        selected=np.load(out/'channel_pred.npy')
        if report['best_epoch']==0:
            assert np.array_equal(selected,reference[:,0])
            assert all(bool((v==0).all()) for n,v in selected_cp['state_dict'].items() if '.extra_project.' in n or '.extra_output.' in n)
        else:
            assert np.array_equal(selected,updated)
            assert all(torch.equal(v,checkpoint['state_dict'][n]) for n,v in selected_cp['state_dict'].items())
        combined=reference.copy();combined[:,0]=selected
        updated_combined=reference.copy();updated_combined[:,0]=updated
        scores=base.metric(truth,combined);updated_scores=base.metric(truth,updated_combined)
        assert abs(scores['TOTAL']-report['reference_replacement_total'])<1e-9
        assert abs(updated_scores['TOTAL']-report['best_after_update']['reference_replacement_total'])<1e-9
        assert np.array_equal(combined[:,1:],reference[:,1:])
        atomic_array(STUDY/f'assembled_{mode}_selected.npy',combined)
        atomic_array(STUDY/f'assembled_{mode}_updated.npy',updated_combined)
        full_update=read_json(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5/fold_5_report.json')
        row=dict(mode=mode,scores=scores,post_update_scores=updated_scores,best_epoch=report['best_epoch'],
            best_updated_epoch=report['best_after_update']['epoch'],stopped_epoch=report['stopped_epoch'],
            total_parameters=report['total_parameters'],trainable_parameters=report['trainable_parameters'],
            selected_delta_reference_TOTAL=scores['TOTAL']-refscore['TOTAL'],
            updated_delta_reference_TOTAL=updated_scores['TOTAL']-refscore['TOTAL'],
            updated_delta_matching_v21_TOTAL=updated_scores['TOTAL']-full_update['best_after_update']['reference_replacement_total'],
            new_updated_checkpoint_disk_replay_all144665_exact=True,
            old_parameters_unchanged_in_trained_and_selected_checkpoints=True,
            selected_checkpoint_replay='same updated checkpoint' if report['best_epoch']>0 else 'epoch0 state verified and original selected prediction exact',
            fixed_PERM_SW_arrays_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,
            post_update_population=breakdown(truth[:,0],reference[:,0],updated,defaults,wells),
            independent_target_parameters=True,replay_seconds=replay_seconds,
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)))
        candidates.append(row);updated_preds[mode]=updated
        print('VERIFIED',json.dumps({k:v for k,v in row.items() if k!='post_update_population'},ensure_ascii=False),flush=True)
        del checkpoint,selected_cp,model,original;gc.collect();torch.cuda.empty_cache()
    best=max(candidates,key=lambda r:r['scores']['TOTAL']);adopt=best['scores']['TOTAL']>refscore['TOTAL']+1e-9
    # The retained current recipe is read from the completed v21 record.
    recipe=read_json(ROOT/'results/deep_learning/champion_scaling_v21_20260928/CURRENT_RECIPE.json')
    if adopt:
        choices=[dict(c) for c in recipe['choices']]
        for c in choices:
            if c['target']=='POR':c.update(checkpoint=best['checkpoint'],model_module='frozen_growth_v22',builder='make_model(mode, master)',
                mode=best['mode'],target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
        recipe['choices']=choices;recipe['scores']=best['scores']
    recipe.update(selected_new_POR=adopt,best_v22_candidate=best['mode'],
        actual_adopted_expansion_factor=int(best['mode'][-1]) if adopt else 1,
        study='frozen_growth_v22_20260928',complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    result=dict(fold=5,rows=144665,reference_scores=refscore,candidates=candidates,
        selected_scores=recipe['scores'],selected_new_POR=adopt,
        post_update_4x_minus_2x=breakdown(truth[:,0],updated_preds['ffn2'],updated_preds['ffn4'],defaults,wells),
        all_updated_checkpoints_disk_replayed=True,all_old_states_unchanged=True,
        fixed_other_targets_reused=True,original_population_order_verified=True,
        seconds=time.perf_counter()-started,finished_at=datetime.now().astimezone().isoformat(),
        complete_five_folds=False,remaining_folds_started=False)
    atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('FINAL_SELECTION',json.dumps(dict(scores=recipe['scores'],new_POR=adopt),ensure_ascii=False))
    print('POPULATIONS',json.dumps({r['mode']:{k:v for k,v in r['post_update_population'].items() if k!='per_well'} for r in candidates},ensure_ascii=False))


if __name__=='__main__':main()

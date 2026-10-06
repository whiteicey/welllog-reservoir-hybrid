"""Post-training implementation checks and winner reload, without new training.
Preserves original reports. Diagnostics never select predictions by row or well.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,time
import torch
import numpy as np
from dual8222_component_query_v7 import ComponentQueryReadout,Network
from prepare_dual8222_query_v7 import trainers

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_component_query_v7_20260928'
PREFIX='pure_dl_dual8222_v7_query_'
PARENT=ROOT/'outputs/pure_dl_dual8222_v5_location_point3280_fold5'
MODES=['center','local']
ADAPTER_PREFIX='0.0.component_readout.'


def fresh(local):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261010)
        return ComponentQueryReadout(local=local).cpu()


def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2)
    reports={}
    for mode in MODES:
        p=ROOT/'outputs'/(PREFIX+mode+'80_fold5')/'fold_5_report.json'
        if not p.exists():raise RuntimeError('Both experiments must complete before post-run verification')
        reports[mode]=json.loads(p.read_text(encoding='utf-8'))
    if trainers():raise RuntimeError('Do not run extra inference while project trainers are active')
    states=[fresh(local).state_dict() for local in [False,True]]
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    count=sum(p.numel() for p in fresh(False).parameters())
    assert count==14081 and all(r['trainable_parameters']==count for r in reports.values())
    diagnostics={}
    for mode in MODES:
        folder=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        last=torch.load(folder/'last_state.pt',map_location='cpu',weights_only=False)
        actual={k[len(ADAPTER_PREFIX):]:v for k,v in last['current_heads'].items() if k.startswith(ADAPTER_PREFIX)}
        module=fresh(mode=='local').eval();module.load_state_dict(actual,strict=True)
        init=states[0]
        movement={}
        for label,keys in [('query_projection',['q_proj.weight','q_proj.bias']),('component_embeddings',['component_embedding']),
                           ('descriptor_projection',['describe.weight','describe.bias']),('output_projection',['output.weight','output.bias'])]:
            movement[label]=float(sum((actual[k].double()-init[k].double()).square().sum() for k in keys).sqrt())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260928)
            features=torch.randn(2,32,2,128)
            descriptors=torch.randn(2,32,2,8,3)
            valid=torch.ones(2,32,dtype=torch.bool);valid[0,:4]=False;valid[1,27:]=False
            with torch.no_grad():
                normal=module(features,descriptors,valid)
                changed=features.clone();changed[~valid]=torch.randn_like(changed[~valid])*1000
                changed_desc=descriptors.clone();changed_desc[~valid]=torch.randn_like(changed_desc[~valid])*1000
                padded=module(changed,changed_desc,valid)
                blank=module(features,descriptors,torch.zeros_like(valid))
                allvalid=torch.ones_like(valid)
                full=module(features,descriptors,allvalid)
                far=features.clone();far[:,20:]+=100
                beyond=module(far,descriptors,allvalid)
                neighbor=features.clone();neighbor[:,11]+=7
                neighbor_result=module(neighbor,descriptors,allvalid)
            assert torch.isfinite(normal).all() and torch.isfinite(blank).all()
            assert torch.count_nonzero(blank)==0 and torch.count_nonzero(normal[~valid])==0
            assert torch.equal(normal[valid],padded[valid]),'Padding leaked into valid component output'
            assert torch.equal(full[:,:4],beyond[:,:4]),'New query module read beyond radius8'
            neighbor_effect=float((full[:,10]-neighbor_result[:,10]).abs().max())
            if mode=='center':assert neighbor_effect==0,'Center control read neighboring feature'
        history=json.loads((folder/'fold_5_validation_history.json').read_text(encoding='utf-8'))
        peak=max(history,key=lambda x:x['official_total'])
        diagnostics[mode]={'last_epoch':last['epoch'],'selected_epoch':reports[mode]['best_epoch'],
            'last_vs_initial_parameter_L2':movement,'nonzero_epoch_peak':{'epoch':peak['epoch'],'TOTAL':peak['official_total']},
            'mask_checks':{'all_padding_finite_and_zero':True,'padding_values_do_not_affect_valid_queries':True,
                           'direct_radius8_enforced':True,'neighbor_effect_at_distance1':neighbor_effect},
            'scope':'synthetic CPU tests using final trained module (not zero-output initial best); parameter movement is not accuracy evidence'}
        del last,module
    old_recipe=json.loads((ROOT/'results/deep_learning/dual8222_location_v5_20260928/CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    baseline=old_recipe['scores']['TOTAL']
    winner=max(MODES,key=lambda m:reports[m]['scores']['TOTAL'])
    proof=None
    if reports[winner]['scores']['TOTAL']>baseline+1e-10:
        from train_dual8222_heads80 import SourceEvaluator,prepare_data,SOURCE
        from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
        from gpu_job_guard_v2 import check_slot
        print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
        start=time.perf_counter();torch.set_num_threads(3)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        folder=ROOT/'outputs'/(PREFIX+winner+'80_fold5')
        cp=torch.load(folder/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
        assert cp['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        master=torch.load(SOURCE,map_location='cpu',weights_only=False)
        data=prepare_data(master)
        numeric,condition=calibration_indices(master['selected_feature_names'])
        original=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        original.load_state_dict(master['state_dict'],strict=True);original.cuda().eval().requires_grad_(False)
        model=Network(original,cp['initialization_manifest'],cp['readout']['mode']).cuda()
        model.heads.load_state_dict(cp['readout']['head_state_dict'],strict=True);model.eval().requires_grad_(False)
        evaluator=SourceEvaluator(data,master)
        with torch.no_grad():prediction,scores=evaluator(model)
        assert np.array_equal(prediction,np.load(folder/'fold_5_pred.npy'))
        assert np.array_equal(prediction[:,1:],np.load(PARENT/'fold_5_pred.npy')[:,1:])
        assert all(abs(scores[c]-reports[winner]['scores'][c])<1e-10 for c in scores)
        # One clearly-labelled intervention, never used to modify the selected recipe.
        module=model.heads[0][0].component_readout
        previous_flag=module.local;module.local=not previous_flag
        with torch.no_grad():intervened,intervened_scores=evaluator(model)
        module.local=previous_flag
        proof={'mode':winner,'different_elements':0,'all144665rows_reloaded_exactly':True,'scores':scores,
               'memory_toggle_intervention':{'trained_weights_fixed':True,'alternate_local':not previous_flag,
                    'scores':intervened_scores,'normal_minus_altered_TOTAL':scores['TOTAL']-intervened_scores['TOTAL'],
                    'meaning':'inference intervention on one trained model, not retraining or independent testing'},
               'seconds_setup_and_two_replays':time.perf_counter()-start}
        (folder/'checkpoint_reload_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        recipe={'mode':winner,'scores':scores,'readout_checkpoint':str((folder/'readout_checkpoint.pt').relative_to(ROOT)),
                'source_backbone':str(SOURCE.relative_to(ROOT)),'model_module':'dual8222_component_query_v7',
                'initialization_manifest_embedded':True,'reloaded_exactly':True,'development_selection':True,
                'not_submission_replacement':True,'extra_folds':False,'new_v7_model_adopted':True}
    else:
        recipe={**old_recipe,'new_v7_model_adopted':False,'decision':'Both completed query heads failed to improve the source; preserve verifiedv5.',
                'reload_note':'No repeated source reload because no improved candidate exists.'}
    result={'observed_at':datetime.now().astimezone().isoformat(),'exact_same_parameter_count':count,
            'exact_same_initial_adapter_state':True,'runs':diagnostics,'winner_reload':proof,
            'new_training':False,'extra_folds':False}
    (STUDY/'implementation_and_reload.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (STUDY/'CURRENT_RECIPE.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'diagnostics':result,'recipe':recipe},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

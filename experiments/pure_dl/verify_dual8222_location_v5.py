"""After the bounded three-arm study, reload the best fixed readout from disk.
Also report one zero-residual inference intervention; never use it to select rows
or replace the chosen model. No training, hyperparameter search, or extra folds.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from torch.nn import functional as F
from train_dual8222_heads80 import SourceEvaluator,prepare_data,SOURCE,atomic_json
from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
from dual8222_location_readout_v5 import Network
from gpu_job_guard_v2 import check_slot
from prepare_dual8222_location_v5 import active_trainers

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_location_v5_20260928'
PREFIX='pure_dl_dual8222_v5_location_'
MODES=['linear','point32','local32']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    reports={}
    for mode in MODES:
        p=ROOT/'outputs'/(PREFIX+mode+'80_fold5')/'fold_5_report.json'
        if not p.exists():raise RuntimeError('All3 reports must exist before best selection: '+mode)
        reports[mode]=json.loads(p.read_text(encoding='utf-8'))
    if active_trainers():raise RuntimeError('Run reload check after trainers have exited')
    winner=max(MODES,key=lambda mode:(reports[mode]['scores']['TOTAL'],-reports[mode]['trainable_parameters']))
    folder=ROOT/'outputs'/(PREFIX+winner+'80_fold5')
    dest=folder/'checkpoint_reload_verification.json'
    if dest.exists():
        print('EXISTING_VERIFICATION',dest.read_text(encoding='utf-8'));return
    print('VERIFY_MODE',winner,flush=True)
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    start=time.perf_counter()
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    cp=torch.load(folder/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
    assert cp['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    data=prepare_data(master)
    numeric,conditioning=calibration_indices(master['selected_feature_names'])
    original=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=conditioning)
    original.load_state_dict(master['state_dict'],strict=True)
    original.cuda().eval().requires_grad_(False)
    model=Network(original,cp['initialization_manifest'],cp['readout']['mode']).cuda()
    model.heads.load_state_dict(cp['readout']['head_state_dict'],strict=True)
    model.eval().requires_grad_(False)
    sums={'count':0,'absolute_sum':0.,'relative_sum':0.,'maximum_absolute':0.,'maximum_relative':0.,'near_bound_count':0}
    def inspect_delta(_module,inputs,output):
        _h,valid,raw_scale=inputs
        sd=F.softplus(raw_scale.detach().float())+.025
        allowed=valid[:,None,:].expand_as(output)
        absolute=output.detach().abs()[allowed]
        ratio=(output.detach().abs()/sd)[allowed]
        sums['count']+=absolute.numel();sums['absolute_sum']+=float(absolute.sum())
        sums['relative_sum']+=float(ratio.sum())
        if absolute.numel():
            sums['maximum_absolute']=max(sums['maximum_absolute'],float(absolute.max()))
            sums['maximum_relative']=max(sums['maximum_relative'],float(ratio.max()))
            sums['near_bound_count']+=int((ratio>.2375).sum())
    residuals=[h.location_residual for h in model.heads[0] if hasattr(h,'location_residual')]
    hooks=[m.register_forward_hook(inspect_delta) for m in residuals]
    evaluator=SourceEvaluator(data,master)
    with torch.no_grad():prediction,scores=evaluator(model)
    for h in hooks:h.remove()
    saved=np.load(folder/'fold_5_pred.npy')
    assert np.array_equal(prediction,saved),'Serialized readout did not replay saved best'
    assert all(abs(scores[t]-reports[winner]['scores'][t])<1e-10 for t in ['POR','PERM','SW','TOTAL'])
    intervention=None
    if residuals:
        zero_hooks=[m.register_forward_hook(lambda _m,_i,out:torch.zeros_like(out)) for m in residuals]
        with torch.no_grad():zero_prediction,zero_scores=evaluator(model)
        for h in zero_hooks:h.remove()
        assert np.array_equal(zero_prediction[:,1:],prediction[:,1:])
        intervention={'scope':'inference-only removal on the trained winner; not a matched retraining ablation or a new selected model',
                      'zero_residual_scores':zero_scores,'normal_minus_zero_TOTAL':scores['TOTAL']-zero_scores['TOTAL'],
                      'changed_POR_predictions':int(np.count_nonzero(zero_prediction[:,0]!=prediction[:,0]))}
    statistics=None
    if sums['count']:
        statistics={'scope':'all valid context positions of validation windows, including repeated context; not only labelled central rows',
                    'n_component_positions':sums['count'],'mean_abs_standardized_location_delta':sums['absolute_sum']/sums['count'],
                    'max_abs_standardized_location_delta':sums['maximum_absolute'],
                    'mean_abs_delta_over_fixed_std':sums['relative_sum']/sums['count'],
                    'max_abs_delta_over_fixed_std':sums['maximum_relative'],
                    'fraction_above95percent_of_bound':sums['near_bound_count']/sums['count'],'bound_fraction':.25}
        assert statistics['max_abs_delta_over_fixed_std']<=.250001
    verify={'observed_at':datetime.now().astimezone().isoformat(),'mode':winner,'scores':scores,
            'different_elements':0,'all144665rows_exact':True,'reload_from_serialized_readout':True,
            'source_sha256_verified':True,'residual_statistics':statistics,'zero_residual_intervention':intervention,
            'seconds_setup_plus_replays':time.perf_counter()-start,'new_training':False,'extra_folds':False}
    atomic_json(dest,verify)
    recipe={'mode':winner,'scores':scores,'readout_checkpoint':str((folder/'readout_checkpoint.pt').relative_to(ROOT)),
            'source_backbone':str(SOURCE.relative_to(ROOT)),'model_module':'dual8222_location_readout_v5',
            'initialization_manifest_embedded':True,'readout_state_key':'readout.head_state_dict',
            'runtime_requires_new_readout_module':True,'reloaded_exactly':True,
            'development_selection':True,'not_submission_replacement':True,'extra_folds':False}
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('RELOADED_AND_VERIFIED',json.dumps(verify,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""Final v24 disk replay and population diagnostics, no training or new selection rules."""
from datetime import datetime
from pathlib import Path
import gc,hashlib,json,sys,time
import numpy as np
import torch
from expert_supervision_v24 import Network,ROOT,STUDY,REFERENCE,MODES,load_data
from evaluate_scratch_scaling_v23 import PopulationEvaluator,paired
from train_dual8222_heads80 import atomic_json,atomic_array
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Final GPU replay requires completed trainers')
    for mode in MODES:
        if not (ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{mode}_fold5/fold_5_report.json').exists():
            raise RuntimeError('Unfinished '+mode)
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    start=time.perf_counter();data,master=load_data()
    reference=np.load(REFERENCE/'fold_5_pred.npy');ids=data['validation']
    truth=data['truth'][ids];wells=data['wells'][ids];defaults=base.dmask(truth)[:,0]
    assert truth.shape==reference.shape==(144665,3)
    with np.load(REFERENCE/'oof.npz') as z:
        assert np.array_equal(truth,z['truth'][z['fold']==4]) and np.array_equal(wells,z['well'][z['fold']==4])
    refscore=base.metric(truth,reference)
    validation_eval=PopulationEvaluator(data,master,ids,reference)
    training_eval=PopulationEvaluator(data,master,np.flatnonzero(data['fold']!=4),reference)
    candidates=[];predictions={}
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_expert_supervision_v24_{mode}_fold5'
        checkpoint=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        for name,digest in checkpoint['config']['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        model=Network(checkpoint['metadata'],mode).cuda();model.load_state_dict(checkpoint['state_dict'],strict=True)
        tick=time.perf_counter();pred,validation=validation_eval(model)
        assert np.array_equal(pred,np.load(out/'channel_pred.npy'))
        assert abs(validation['POR']-report['target_score'])<1e-8
        _,fit=training_eval(model)
        combined=reference.copy();combined[:,0]=pred
        assert np.array_equal(combined[:,1:],reference[:,1:])
        score=base.metric(truth,combined)
        assert abs(score['TOTAL']-report['reference_replacement_total'])<1e-8
        atomic_array(STUDY/f'assembled_{mode}_pred.npy',combined)
        row=dict(mode=mode,scores=score,delta_reference_TOTAL=score['TOTAL']-refscore['TOTAL'],
            best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],parameter_count=report['trainable_parameters'],
            validation=validation,fit=fit,comparison_to_champion=paired(truth[:,0],reference[:,0],pred,defaults,wells),
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)),
            all144665_predictions_disk_reloaded_exactly=True,original_row_order_verified=True,
            fixed_PERM_SW_arrays_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,
            independent_target_parameters=True,sources_unchanged=True,
            train_and_validation_inference_seconds=time.perf_counter()-tick)
        candidates.append(row);predictions[mode]=pred
        print('VERIFIED',json.dumps({k:v for k,v in row.items() if k not in ('fit','validation','comparison_to_champion')},ensure_ascii=False),flush=True)
        print('POPULATIONS',mode,json.dumps({'fit':{k:v for k,v in fit.items() if k!='per_well'},
            'validation':{k:v for k,v in validation.items() if k!='per_well'}},ensure_ascii=False),flush=True)
        del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    comparison=paired(truth[:,0],predictions['mixture'],predictions['independent'],defaults,wells)
    best=max(candidates,key=lambda c:c['scores']['TOTAL'])
    adopt=best['scores']['TOTAL']>refscore['TOTAL']+1e-9
    prior=json.loads((ROOT/'results/deep_learning/scratch_scaling_v23_20260928/CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    choices=[dict(c) for c in prior['choices']]
    if adopt:
        for c in choices:
            if c['target']=='POR':c.update(key='POR_v24_'+best['mode'],mode=best['mode'],architecture='private_dual',
                checkpoint=best['checkpoint'],model_module='expert_supervision_v24',builder='Network(checkpoint[metadata],mode)',
                target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected=best['scores'] if adopt else refscore
    result=dict(fold=5,rows=144665,reference_scores=refscore,candidates=candidates,
        supervision_comparison=comparison,selected_scores=selected,selected_new_POR=adopt,
        all_new_checkpoints_disk_replayed=True,old_other_targets_reused=True,
        old_other_target_replay_provenance='results/deep_learning/latent_gate_v17_20260928/assembled_report.json',
        diagnostic_evaluator_sha256=hashlib.sha256(Path(__file__).with_name('evaluate_scratch_scaling_v23.py').read_bytes()).hexdigest(),
        seconds=time.perf_counter()-start,finished_at=datetime.now().astimezone().isoformat(),
        complete_five_folds=False,remaining_folds_started=False)
    recipe=dict(scores=selected,choices=choices,selected_new_POR=adopt,best_v24_candidate=best['mode'],
        source_preprocessing='results/deep_learning/champion_scaling_v21_20260928/data_manifest.json',
        complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('SUPERVISION_DELTA',json.dumps({k:v for k,v in comparison.items() if k!='per_well'},ensure_ascii=False))
    print('SELECTED',json.dumps(recipe,ensure_ascii=False))


if __name__=='__main__':main()

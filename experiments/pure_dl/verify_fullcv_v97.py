"""Disk replay and original-population audit for completed fixed-recipe folds.

Each immutable fold proof is reusable by content hash. The full OOF is certified
only after all original rows are finite, all four disk replays pass, and the
verified fifth is unchanged. Never compute a partial OOF as a complete result.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from fullcv_context_v97 import ROOT, STUDY, RECIPE, original_population, validate_metadata_source, array_sha
from fullcv_models_v97 import fold_dir, root_dir, root_path, best_path, make_model
from train_stage_fullcv_v97 import Evaluator
import train_invariant as base


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_signature(fold):
    # Selected dependencies and preprocessing cache only. Completed recovery
    # states and observations are deliberately outside this inference contract.
    files=[root_path(fold),fold_dir(fold)/'data.pt',fold_dir(fold)/'amplitude.npz',
        fold_dir(fold)/'selected_factors.pt',fold_dir(fold)/'private_POR.pt',fold_dir(fold)/'private_PERM.pt',
        fold_dir(fold)/'fold_pred.npy']
    files+=list(fold_dir(fold).glob('*/model_best.pt'))
    files+=[Path(__file__),Path(__file__).with_name('train_stage_fullcv_v97.py'),Path(__file__).with_name('fullcv_models_v97.py')]
    assert all(p.exists() for p in files)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def verify_fold(fold,pop):
    out=fold_dir(fold);proof_path=out/'disk_replay_proof.json'
    assert (out/'fold_report.json').exists(),'The final three-channel fold must finish before disk replay'
    signature=source_signature(fold)
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        if proof['source_sha256']==signature:
            assert proof['all_disk_predictions_exact']
            print('REUSE_VERIFIED_DISK_REPLAY',fold,flush=True);return proof
        raise RuntimeError('Immutable completed fold/proof source changed; investigate instead of silently replacing')
    master=torch.load(root_path(fold),map_location='cpu',weights_only=False)
    rows=validate_metadata_source(master,pop,fold-1)
    contract=base.feature_contract();assert len(contract['whitelist'])==206 and master['selected_feature_names']==contract['whitelist']
    assert master['config']['seed']==20261006 and master['config']['original_fold']==fold
    assert master['config']['epochs']==80 and master['config']['steps']==64
    assert master['config']['early_stopping']==dict(patience=18,minimum_epochs=10,maximum_epochs=80,criterion='official_TOTAL',mode='max')
    defaults=base.dmask(pop['truth'])
    for c in range(3):
        transformed=base.TRANSFORM[c](pop['truth'][:,c]);fit=transformed[(pop['fold']!=fold-1)&~defaults[:,c]]
        q25,q50,q75=np.percentile(fit,[25,50,75])
        assert master['regression_median'][c]==q50 and master['regression_scale'][c]==max(float(q75-q25),.1)
    cache=torch.load(out/'data.pt',map_location='cpu',weights_only=False);assert cache['root_sha256']==sha(root_path(fold))
    data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache['data'].items()}
    metadata={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache['metadata'].items()};del cache
    assert np.array_equal(data['truth'],pop['truth']) and np.array_equal(data['fold'],pop['fold']) and np.array_equal(data['wells'],pop['wells'])
    assert np.array_equal(data['validation'],rows['validation'])
    assert np.array_equal(data['train_raw'].cpu().numpy(),rows['training_raw'])
    validation_raw=torch.as_tensor(rows['validation_raw'],device='cuda')
    assert not bool(data['labeled'][validation_raw].any()) and bool((data['labels'][validation_raw]==0).all())
    assert bool((data['quality'][validation_raw]==1).all())
    bank_train=metadata['bank_manifest']['train_wells'];assert bank_train==rows['train_wells']
    assert torch.equal(metadata['reference_inputs'],data['well_bank'][torch.as_tensor(bank_train,device='cuda')].mean(0))
    with np.load(out/'amplitude.npz') as z:side=torch.as_tensor(z['amplitude'],device='cuda')
    features=data['x'];amplitude=torch.cat((features,side),-1);predictions=[];source_reports=[]
    expected=np.load(out/'fold_pred.npy');assert expected.shape==(len(rows['validation']),3)
    for c,stage in enumerate(('por_anchored','perm_split','sw_density')):
        data['x']=amplitude if c<2 else features
        model,spec,trace=make_model(fold,stage,master,metadata);model.cuda()
        cp=torch.load(best_path(fold,stage),map_location='cpu',weights_only=False)
        assert cp['original_fold']==fold and cp['target']==base.TS[c] and cp['source_sha256']==sha(root_path(fold))
        if c<2:model.model.heads[spec[2]].load_state_dict(cp['delta_state'],strict=True)
        else:model.model.load_state_dict(cp['state_dict'],strict=True)
        result,score=Evaluator(data,master,c,spec[4])(model)
        assert np.array_equal(result,expected[:,c]),(fold,stage,float(np.abs(result-expected[:,c]).max()))
        assert np.array_equal(result,np.load(best_path(fold,stage).parent/'channel_pred.npy'))
        assert abs(score-cp['target_score'])<1e-9
        predictions.append(result);source_reports.append(dict(target=base.TS[c],checkpoint=str(best_path(fold,stage).relative_to(ROOT)),best_epoch=cp['best_epoch'],score=score,dependencies=trace))
        if hasattr(model.model,'close'):model.model.close()
        del model,cp;torch.cuda.empty_cache()
    assembled=np.column_stack(predictions);assert np.array_equal(assembled,expected)
    scores=base.metric(pop['truth'][rows['validation']],assembled)
    recorded=json.loads((out/'fold_report.json').read_text());assert recorded['scores']==scores and recorded['rows']==len(assembled)
    proof=dict(fold=fold,rows=len(assembled),train_wells=rows['train_wells'],validation_wells=rows['heldout_wells'],
        training_raw_sha256=array_sha(rows['training_raw']),validation_raw_sha256=array_sha(rows['validation_raw']),
        all_disk_predictions_exact=True,all_original_rows_once=True,preprocessing_training_population_verified=True,
        validation_labels_unallocated_for_loss=True,scores=scores,source_models=source_reports,source_sha256=signature,
        verified_at=datetime.now().astimezone().isoformat())
    proof_path.write_text(json.dumps(proof,indent=2),encoding='utf-8');print('VERIFIED_FOLD_DISK',json.dumps({k:v for k,v in proof.items() if k not in ('source_sha256','source_models')}),flush=True)
    return proof


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3,4]);ap.add_argument('--final',action='store_true');a=ap.parse_args()
    assert a.fold is not None or a.final
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    pop=original_population();protocol=json.loads((STUDY/'protocol.json').read_text())
    assert protocol['data_population_sha256']==dict(truth=array_sha(pop['truth']),wells=array_sha(pop['wells']),fold=array_sha(pop['fold']),raw_rows=array_sha(pop['valid_to_raw']))
    if a.fold is not None:verify_fold(a.fold,pop)
    if not a.final:return
    final=json.loads((STUDY/'FINAL_REPORT.json').read_text());assert final['complete_five_folds']
    assert sha(RECIPE)==protocol['selected_recipe_sha256']
    promotion=json.loads((STUDY/'promotion.json').read_text());assert sha(STUDY/'oof_pending.npz')==promotion['partial_oof_sha256']
    with np.load(STUDY/'oof_progress.npz') as z:truth=z['truth'];pred=z['pred'];fold=z['fold'];well=z['well']
    assert np.array_equal(truth,pop['truth']) and np.array_equal(fold,pop['fold']) and np.array_equal(well,pop['wells'])
    assert pred.shape==(723556,3) and np.isfinite(pred).all()
    with np.load(STUDY/'oof_pending.npz') as z:fifth=z['pred'][fold==4]
    assert np.array_equal(pred[fold==4],fifth) and base.metric(truth[fold==4],fifth)==promotion['scores']
    proofs=[verify_fold(f,pop) for f in (1,2,3,4)]
    for f in (1,2,3,4):assert np.array_equal(pred[fold==f-1],np.load(fold_dir(f)/'fold_pred.npy'))
    scores=base.metric(truth,pred);assert scores==final['oof_scores']
    assert [r['fold'] for r in final['folds']]==[1,2,3,4,5]
    evidence=dict(complete_five_folds=True,rows=723556,wells=80,original_row_order=True,all_four_disk_replays_passed=True,
        fifth_reused_exact=True,source_recipe_sha256=sha(RECIPE),oof_sha256=sha(STUDY/'oof_progress.npz'),oof_scores=scores,
        strict_oof82_passed=scores['TOTAL']>82,fold_proof_files=[str((fold_dir(f)/'disk_replay_proof.json').relative_to(ROOT)) for f in (1,2,3,4)],
        verified_at=datetime.now().astimezone().isoformat())
    (STUDY/'OOF_VERIFICATION.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print('VERIFIED_FULL_OOF',json.dumps(evidence),flush=True)


if __name__=='__main__':main()

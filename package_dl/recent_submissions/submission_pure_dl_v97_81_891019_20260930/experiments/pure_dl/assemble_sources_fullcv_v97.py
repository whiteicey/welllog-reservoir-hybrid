"""Fixed whole-target source exports and final original-fold assembly."""
import argparse
import json
import numpy as np
import torch
from fullcv_context_v97 import ROOT, STUDY, sha
from fullcv_models_v97 import fold_dir, root_path, best_path, constructor_bindings
from fullcv_resume_v97 import atomic_save
from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
from dual8222_factorized_readout import Network as Factors
from independent_channels_v8 import Network as Private
import train_invariant as base


def original_model(master):
    numeric,conditioning=calibration_indices(master['selected_feature_names'])
    m=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=conditioning)
    m.load_state_dict(master['state_dict'],strict=True);return m


def identity_proof():
    master=torch.load(ROOT/'outputs/pure_dl_paired_local_context_router80_sequence/fold_5.pt',map_location='cpu',weights_only=False)
    previous=torch.load(ROOT/'outputs/pure_dl_dual8222_v5_location_point3280_fold5/readout_checkpoint.pt',map_location='cpu',weights_only=False)
    proof=[]
    for t in ('POR','PERM'):
        path=ROOT/'outputs'/f'pure_dl_independent_v8_{t}_dual80_fold5/model_best.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False);model=Private(master,previous,t,'dual');state=model.state_dict()
        assert cp['best_epoch']==0 and state.keys()==cp['state_dict'].keys()
        assert all(torch.equal(v,cp['state_dict'][k]) for k,v in state.items())
        proof.append(dict(target=t,best_epoch=0,all_state_tensors_exact=True,checkpoint=str(path.relative_to(ROOT)),sha256=sha(path)))
    (STUDY/'identity_stage_proof.json').write_text(json.dumps(dict(proof=proof,
        protocol='v8 selected initial private extraction is a fixed identity stage for all remaining folds; no new v8 finetuning search'),indent=2),encoding='utf-8')
    print('IDENTITY_PROOF',json.dumps(proof))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3,4]);ap.add_argument('--action',choices=['identity-proof','factors','private','final'],required=True);a=ap.parse_args()
    if a.action=='identity-proof':identity_proof();return
    f=a.fold;assert f is not None;out=fold_dir(f);master=torch.load(root_path(f),map_location='cpu',weights_only=False)
    if a.action=='factors':
        m=Factors(original_model(master),'gate_only')
        por=torch.load(best_path(f,'por_density'),map_location='cpu',weights_only=False)
        perm=torch.load(best_path(f,'perm_gate'),map_location='cpu',weights_only=False)
        states=[por['head_state_dict'],perm['head_state_dict'],{k:v.detach().cpu().clone() for k,v in m.heads[2].state_dict().items()}]
        for cp in (por,perm):assert cp['original_fold']==f and cp['source_sha256']==sha(root_path(f))
        atomic_save(dict(head_state_dicts=states,source_kinds=['target_group']*3,original_fold=f,
            source_checkpoint=str(root_path(f).relative_to(ROOT)),source_sha256=sha(root_path(f))),out/'selected_factors.pt')
    elif a.action=='private':
        proof=json.loads((STUDY/'identity_stage_proof.json').read_text());assert all(p['all_state_tensors_exact'] for p in proof['proof'])
        previous=torch.load(best_path(f,'por_point'),map_location='cpu',weights_only=False)
        assert previous['original_fold']==f
        for t in ('POR','PERM'):
            m=Private(master,previous,t,'dual')
            atomic_save(dict(state_dict=m.state_dict(),best_epoch=0,original_fold=f,target=t,identity_stage=True,
                source_checkpoint=str(root_path(f).relative_to(ROOT)),source_sha256=sha(root_path(f))),out/f'private_{t}.pt')
    else:
        with open(out/'data_manifest.json',encoding='utf-8') as stream:manifest=json.load(stream)
        pending=np.load(STUDY/'oof_pending.npz');truth=pending['truth'];fold=pending['fold'];well=pending['well']
        old=STUDY/'oof_progress.npz';pred=np.load(old)['pred'].copy() if old.exists() else pending['pred'].copy()
        stages=('por_anchored','perm_split','sw_density');columns=[];reports=[]
        for stage in stages:
            p=best_path(f,stage);r=json.loads((p.parent/'report.json').read_text())
            assert r['fold']==f and r['best_predictions_reproduced_exactly'] and r['source_sha256']==sha(root_path(f))
            columns.append(np.load(p.parent/'channel_pred.npy'));reports.append(r)
        assembled=np.column_stack(columns);assert assembled.shape==(int((fold==f-1).sum()),3) and np.isfinite(assembled).all()
        pred[fold==f-1]=assembled;scores=base.metric(truth[fold==f-1],assembled)
        np.save(out/'fold_pred.npy',assembled)
        report=dict(fold=f,scores=scores,rows=len(assembled),source_models=reports,complete_five_folds=False)
        (out/'fold_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with (STUDY/'oof_progress.tmp').open('wb') as stream:np.savez_compressed(stream,truth=truth,pred=pred,fold=fold,well=well)
        (STUDY/'oof_progress.tmp').replace(STUDY/'oof_progress.npz')
        completed=[i+1 for i in range(5) if np.isfinite(pred[fold==i]).all()]
        status=dict(completed_folds=completed,pending_folds=[i for i in range(1,6) if i not in completed],complete_five_folds=len(completed)==5)
        if status['complete_five_folds']:
            assert np.isfinite(pred).all() and len(pred)==723556
            fold_reports=[json.loads((fold_dir(i)/'fold_report.json').read_text()) for i in (1,2,3,4)]
            fifth=json.loads((STUDY/'promotion.json').read_text());fold_reports.append(dict(fold=5,scores=fifth['scores'],rows=fifth['rows'],reused_verified_fifth=True))
            status.update(oof_scores=base.metric(truth,pred),folds=fold_reports,original_row_order=True)
            status['strict_oof82_passed']=status['oof_scores']['TOTAL']>82
            (STUDY/'FINAL_REPORT.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        (STUDY/'training_progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('ASSEMBLED_FOLD',json.dumps(report|dict(progress=status)),flush=True)


if __name__=='__main__':main()

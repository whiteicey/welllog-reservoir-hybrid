"""Promote one completed fixed-fifth recipe; verify raw labels and cache alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fifth',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();folder=args.fifth.resolve();out=args.out.resolve()
    assert not (out/'promotion.json').exists(),'Refuse to overwrite a promotion'
    report=json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))
    runtime=json.loads((folder/'runtime_status.json').read_text(encoding='utf-8'))
    assert runtime['status']=='complete' and report['fold']==5
    assert report['best_checkpoint_predictions_reproduced'] and report['checkpoint_persisted']
    assert report['all_validation_rows_covered_exactly_four_times']
    cp=torch.load(folder/'fold_5.pt',map_location='cpu',weights_only=False)
    cfg=cp['config'];assert cfg['seed']==20261006 and cfg['epochs']==80
    assert cfg['attribution_ablation'] in ('control','no_global_full')
    assert cfg['selection'].startswith('highest fixed ABD')
    rec=base.load_split(base.ROOT.parent/'data','train')
    ys=[frame[base.TS].to_numpy(np.float64) for _,frame in rec]
    valid=[np.isfinite(y).all(1) for y in ys]
    truth=np.concatenate([y[m] for y,m in zip(ys,valid)])
    wells=np.concatenate([np.full(m.sum(),i,np.int32) for i,m in enumerate(valid)])
    starts=np.r_[0,np.cumsum([len(y) for y in ys])]
    raw=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
    fold=np.full(len(truth),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(truth,groups=wells)):fold[ix]=f
    assert truth.shape==(723556,3) and len(rec)==80
    validation=np.flatnonzero(fold==4)
    assert np.array_equal(cp['validation_raw_rows'],raw[validation])
    assert np.array_equal(np.sort(cp['training_raw_rows']),np.sort(raw[fold!=4]))
    pred=np.load(folder/'fold_5_pred.npy')
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    with np.load(folder/'oof.npz') as cache:
        assert np.array_equal(cache['truth'],truth) and np.array_equal(cache['fold'],fold)
        assert np.array_equal(cache['well'],wells) and np.array_equal(cache['pred'][validation],pred)
        assert np.isnan(cache['pred'][fold!=4]).all()
    scores=base.metric(truth[validation],pred)
    assert abs(scores['TOTAL']-report['scores']['TOTAL'])<1e-10
    assert abs(scores['TOTAL']-cp['best_validation_score'])<1e-10
    assert scores['TOTAL']>80.4 and scores['TOTAL']>80.40915186344968
    for name,expected in cfg['dependency_sha256'].items():
        assert sha(base.ROOT/name)==expected,('Trained dependency changed',name)
    for source in cfg['frozen_companions']:
        path=base.ROOT/source['folder']/'fold_5.pt'
        assert sha(path)==source['sha256']
    d_folder=base.ROOT/'outputs/pure_dl_global_cosine_sequence'
    d_checks=[]
    for f in range(5):
        path=d_folder/f'fold_{f+1}.pt';item=torch.load(path,map_location='cpu',weights_only=False)
        rr=json.loads((d_folder/f'fold_{f+1}_report.json').read_text(encoding='utf-8'))
        assert rr['best_checkpoint_predictions_reproduced'] and item['config']['seed']==20261006
        assert np.array_equal(item['validation_raw_rows'],raw[fold==f])
        assert np.array_equal(np.sort(item['training_raw_rows']),np.sort(raw[fold!=f]))
        d_checks.append({'fold':f+1,'path':str(path),'sha256':sha(path)})
    proof={'passed':True,'screening_fold':5,'score':scores['TOTAL'],'scores':scores,
           'baseline_total':80.40915186344968,'ablation':cfg['attribution_ablation'],
           'fifth_folder':str(folder),'fifth_checkpoint_sha256':sha(folder/'fold_5.pt'),
           'fifth_predictions_sha256':sha(folder/'fold_5_pred.npy'),
           'fifth_best_epoch':report['best_epoch'],'fifth_config':cfg,'d_checkpoints':d_checks,
           'frozen_companions_fold5':cfg['frozen_companions'],'validation_rows':len(validation),
           'total_valid_rows':len(truth),'complete_five_folds':False,
           'protocol':'B independent original80 solo-selected; D existing75; A original80 training with frozen B/D and fused validation selection. Equal distributions and fixed decoder. No validation-row oracle or new weights.',
           'scope':'repeated fixed-fifth development selection, not independent test evidence'}
    out.mkdir(parents=True,exist_ok=True)
    base.write(out/'promotion.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('fifth_config','d_checkpoints')},indent=2),flush=True)


if __name__=='__main__':main()

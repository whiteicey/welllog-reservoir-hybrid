"""Assemble only held-out fold predictions and score original complete population."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve()
    proof=json.loads((out/'promotion.json').read_text(encoding='utf-8'))
    assert proof['passed'] and proof['score']>80.4
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
    pred=np.full_like(truth,np.nan);coverage=np.zeros(len(truth),np.int8);reports=[]
    for f in range(5):
        folder=Path(proof['fifth_folder']) if f==4 else out/'A'/f'fold_{f+1}'
        cp=torch.load(folder/f'fold_{f+1}.pt',map_location='cpu',weights_only=False)
        report=json.loads((folder/f'fold_{f+1}_report.json').read_text(encoding='utf-8'))
        assert report['fold']==f+1 and report['best_checkpoint_predictions_reproduced']
        assert cp['config']['seed']==20261006 and cp['config']['epochs']==80
        assert cp['config']['attribution_ablation']==proof['ablation']
        ix=np.flatnonzero(fold==f)
        assert np.array_equal(cp['validation_raw_rows'],raw[ix])
        assert np.array_equal(np.sort(cp['training_raw_rows']),np.sort(raw[fold!=f]))
        value=np.load(folder/f'fold_{f+1}_pred.npy')
        assert value.shape==(len(ix),3) and np.isfinite(value).all()
        with np.load(folder/'oof.npz') as cached:
            assert np.array_equal(cached['truth'],truth) and np.array_equal(cached['fold'],fold)
            assert np.array_equal(cached['well'],wells)
            assert np.array_equal(cached['pred'][ix],value) and np.isnan(cached['pred'][fold!=f]).all()
        score=base.metric(truth[ix],value)
        assert abs(score['TOTAL']-report['scores']['TOTAL'])<1e-10
        assert abs(score['TOTAL']-cp['best_validation_score'])<1e-10
        for j,name in enumerate(base.TS):
            lo,hi=base.BOUNDS[name]
            assert (value[:,j]>=lo-1e-8).all() and (value[:,j]<=hi+1e-8).all()
        for source in cp['config']['frozen_companions']:
            path=base.ROOT/source['checkpoint'] if 'checkpoint' in source else base.ROOT/source['folder']/f'fold_{f+1}.pt'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
            companion=torch.load(path,map_location='cpu',weights_only=False)
            assert np.array_equal(companion['validation_raw_rows'],raw[ix])
            assert np.array_equal(np.sort(companion['training_raw_rows']),np.sort(raw[fold!=f]))
        pred[ix]=value;coverage[ix]+=1
        reports.append({'fold':f+1,'scores':score,'rows':len(ix),'best_epoch':report['best_epoch'],
                        'stopped_epoch':report['stopped_epoch'],'checkpoint_sha256':hashlib.sha256((folder/f'fold_{f+1}.pt').read_bytes()).hexdigest()})
    assert (coverage==1).all() and np.isfinite(pred).all()
    scores=base.metric(truth,pred)
    np.savez_compressed(out/'oof.npz',truth=truth,pred=pred,fold=fold,well=wells)
    report={'complete_five_folds':True,'oof_scores':scores,'folds':reports,'original_rows':723556,
            'wells':80,'validation_coverage_exactly_once':True,'all_companions_same_fold':True,
            'original_labels_and_group_folds_verified':True,'goal_strict_gt82':bool(scores['TOTAL']>82),
            'delta_previous_complete_oof':scores['TOTAL']-81.71323318479078,
            'scope':'Grouped OOF with per-fold validation checkpoint selection; fifth repeatedly used for development. No independent test score.'}
    base.write(out/'report.json',report)
    print('FULL_OOF_VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':main()

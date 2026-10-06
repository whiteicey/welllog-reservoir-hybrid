"""Complete held-out OOF replay of the exported portable predictor, one GPU."""
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
D=ROOT.parent/'submission_pure_dl_v370_82_194651_20261002'
S=ROOT/'results/deep_learning/two_seed_fullcv_v370_20261002'
spec=importlib.util.spec_from_file_location('portable_predict_v370',D/'predict.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from src.data import load_split
from train_invariant import metric

def main():
    p.configure_runtime();started=time.perf_counter();records=load_split(ROOT.parent/'data','train')
    with np.load(S/'OOF_POPULATION.npz') as z:truth=z['truth'];fold=z['fold'];wells=z['wells']
    expected=np.load(S/'diagnostic_OOF.npy');assembled=np.full_like(expected,np.nan);proofs=[]
    for f in range(1,6):
        predictor=p.Predictor(D/'models','cuda',folds=(f,));heldout=np.unique(wells[fold==f-1])
        predictions=predictor.predict_records([records[wi] for wi in heldout])
        for wi,pred in zip(heldout,predictions):
            name,frame=records[wi]
            # Labels determine verification population only; the actual predictor
            # strips outputs and accepts exactly the same input interface as test.
            valid=np.isfinite(frame[list(p.TARGETS)].to_numpy()).all(1);rows=np.flatnonzero(wells==wi)
            assert (fold[rows]==f-1).all() and len(rows)==valid.sum()
            assembled[rows]=pred[valid]
        rows=np.flatnonzero(fold==f-1);different=(assembled[rows]!=expected[rows]).sum(0)
        result=dict(fold=f,rows=len(rows),different_predictions=different.tolist(),scores=metric(truth[rows],assembled[rows]))
        proofs.append(result);print('PACKAGED_FOLD',json.dumps(result),flush=True)
        del predictor;torch.cuda.empty_cache()
    exact=np.array_equal(assembled,expected)
    result=dict(passed=exact,all723556_packaged_predictions_exact=exact,rows=len(truth),folds=proofs,
                scores=metric(truth,assembled),seconds=time.perf_counter()-started,one_GPU=True,
                scope='Full portable single-fold prediction on original16 held-out wells per fold; original rows and per-head tuned ratios, no fitting or selection.')
    np.save(D/'models/packaged_oof.npy',assembled)
    (D/'models/packaged_oof_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert exact,'Portable prediction differs; inspect saved per-fold mismatch evidence before delivery'
    print('PACKAGED_OOF_VERIFIED',json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)

if __name__=='__main__':main()

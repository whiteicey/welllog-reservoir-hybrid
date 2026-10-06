"""Replay packaged graphs and rebuilt inputs against immutable original OOF."""
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
DELIVERY=REPO.parent/'submission_pure_dl_v97_81_891019_20260930'
spec=importlib.util.spec_from_file_location('delivery_predict',DELIVERY/'predict.py')
predict=importlib.util.module_from_spec(spec);spec.loader.exec_module(predict)
from train_stage_fullcv_v97 import Evaluator
from src.data import load_split
import train_invariant as base

def main():
    predict.configure_runtime();started=time.perf_counter();records=load_split(REPO.parent/'data','train')
    with np.load(REPO/'results/deep_learning/fullcv_v97_20260930/oof_progress.npz') as z:
        expected=z['pred'];fold=z['fold'];truth=z['truth']
    assembled=np.full_like(expected,np.nan);reports=[]
    for f in range(1,6):
        predictor=predict.Predictor(DELIVERY/'models','cuda',folds=(f,))
        b=predictor.bundles[f];master=b['preprocessing']
        if f<5:
            cache=torch.load(REPO/f'outputs/pure_dl_fullcv_v97/fold_{f}/data.pt',map_location='cpu',weights_only=False)
            data=cache['data'];del cache
            with np.load(REPO/f'outputs/pure_dl_fullcv_v97/fold_{f}/amplitude.npz') as z:amp=z['amplitude']
        else:
            data=torch.load(REPO/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
            amp=None
            data['population_rows']=dict(heldout_wells=np.unique(data['wells'][data['fold']==4]).tolist())
        wells=data['population_rows']['heldout_wells'];features=data['x']
        amplitudes=[];feature_exact=[]
        for wi,(_,frame) in enumerate(records):
            # Only held-out wells need preprocessing replay; all raw inputs are
            # kept for the original evaluator's global row indexing.
            if wi not in wells:continue
            x,amplitude,depth,absolute,bank=predictor.features(frame,f)
            lo,hi=data['starts'][wi:wi+2]
            assert torch.equal(x.cpu(),features[lo:hi]),(f,wi,'206-feature mismatch')
            assert torch.equal(absolute.cpu(),data['absolute'][lo:hi]),(f,wi,'depth mismatch')
            if amp is not None:assert np.array_equal(amplitude[:,206:].cpu().numpy(),amp[lo:hi]),(f,wi,'amplitude mismatch')
            amplitudes.append((int(lo),int(hi),amplitude[:,206:].cpu()));feature_exact.append(int(wi))
        if amp is None:
            amp=torch.zeros(len(features),63,dtype=features.dtype)
            for lo,hi,a in amplitudes:amp[lo:hi]=a
        else:amp=torch.as_tensor(amp)
        # Well context is deterministic stratified input sampling; no labels.
        starts=data['starts'];sample=np.stack([lo+np.minimum(((np.arange(128)+.5)*(hi-lo)/128).astype(np.int64),hi-lo-1) for lo,hi in zip(starts[:-1],starts[1:])])
        row_well=np.repeat(np.arange(80),np.diff(starts))
        data['well_bank']=features[torch.as_tensor(sample)];data['row_well']=torch.as_tensor(row_well)
        data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        features=data['x'];amplitude=torch.cat((features,amp.cuda()),-1);preds=[]
        for c,model in enumerate(b['models']):
            data['x']=amplitude if c<2 else features
            result,score=Evaluator(data,master,c,1024 if c==2 else 256)(model)
            target_expected=expected[fold==f-1,c]
            assert np.array_equal(result,target_expected),(f,c,'packaged graph prediction differs',int(np.sum(result!=target_expected)))
            preds.append(result)
        combined=np.column_stack(preds);assembled[fold==f-1]=combined
        report=dict(fold=f,rows=len(combined),all_three_packaged_predictions_exact=True,rebuilt_heldout206_features_exact=True,
                    rebuilt_heldout_amplitude_exact_to_cache=f<5,amplitude_fold5_scope='frozen original63-feature recipe replayed' if f==5 else None,
                    heldout_wells=feature_exact,scores=base.metric(truth[fold==f-1],combined))
        reports.append(report);print('PACKAGED_REPLAY',json.dumps(report),flush=True)
        del predictor,b,model,data,features,amplitude,amp;torch.cuda.empty_cache()
    assert np.array_equal(assembled,expected)
    result=dict(passed=True,all723556_packaged_predictions_exact=True,original_oof_scores=base.metric(truth,assembled),
                folds=reports,seconds=time.perf_counter()-started,scope='complete packaged graph replay and heldout input preprocessing; no retraining')
    (DELIVERY/'models/packaged_oof_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PACKAGED_OOF_VERIFIED',json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)

if __name__=='__main__':main()

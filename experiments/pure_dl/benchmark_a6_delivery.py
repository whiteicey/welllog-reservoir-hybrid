"""Verify standalone exported predictor against every saved fixed-fold5 prediction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(args.package.resolve()))
    import predict
    from gpu_job_guard_v2 import check_slot
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    predict.configure_runtime();started=time.perf_counter()
    predictor=predict.Predictor(args.package/'models','cuda')
    raw=np.asarray(predictor.master['validation_raw_rows']);offset=0;pred=[];truth=[];found=[];wells=0
    for well,frame in predict.load_inputs(args.data/'train'):
        rows=np.arange(offset,offset+len(frame));mask=np.isin(rows,raw)
        if mask.any():
            y=frame[predict.TARGETS].to_numpy(np.float64)
            assert np.isfinite(y[mask]).all()
            # Selection comes from saved checkpoint row IDs, never target values.
            pred.append(predictor.predict_well(frame,prediction_rows=np.flatnonzero(mask)));truth.append(y[mask]);found.append(rows[mask]);wells+=1
            print('VALIDATED_WELL',well,int(mask.sum()),flush=True)
        offset+=len(frame)
    assert np.array_equal(np.concatenate(found),raw)
    p=np.concatenate(pred);y=np.concatenate(truth)
    reference=np.load(repo/'outputs/pure_dl_a6_channel_heads_own80_fold5/fold_5_pred.npy')
    accuracy=np.c_[np.maximum(0,1-np.abs(p[:,0]-y[:,0])/(.08*(np.abs(y[:,0])+1e-5))),
        np.maximum(0,1-np.abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
        np.maximum(0,1-np.abs(p[:,2]-y[:,2])/(.05*(np.abs(y[:,2])+1e-5)))]
    scores=accuracy.mean(0)*100
    summary={'wells':wells,'rows':len(p),'packaged_predictions_exactly_equal_saved':bool(np.array_equal(p,reference)),
        'different_elements':int(np.count_nonzero(p!=reference)),'maximum_absolute_difference':float(np.max(np.abs(p-reference))),
        'scores':dict(zip(predict.TARGETS+['TOTAL'],[*scores,float(scores@np.array([.3,.35,.35]))])),
        'seconds':time.perf_counter()-started,'validation_labels_used_for_prediction':False,
        'scope':'Standalone exported model and preprocessing, all original fifth-fold rows; same saved head checkpoints'}
    out=args.package/'models/evidence/packaged_fold5_verification.json'
    out.write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)
    assert summary['packaged_predictions_exactly_equal_saved'], 'Exported inference changed the verified predictions'
    assert abs(summary['scores']['TOTAL']-80.50155201341903)<1e-10


if __name__=='__main__':main()

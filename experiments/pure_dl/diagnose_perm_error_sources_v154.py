"""Describe frozen fold5 PERM errors; no fitting or prediction selection."""
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/perm_ranked_probability_v154_20261001'

def main():
    d=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    ix=d['validation'];truth=d['truth'][ix];y=truth[:,1]
    out=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional'
    r=json.loads((out/'report.json').read_text());p=np.load(out/'channel_pred.npy')
    assert r['predictions_reproduced_exactly'] and p.shape==y.shape and np.isfinite(p).all()
    actual_default=base.dmask(truth)[:,1];pred_default=p==base.DEFAULTS['PERM']
    log_error=np.log10(np.maximum(p/np.maximum(y,1e-5),1e-5))
    score=np.maximum(0,1-np.abs(log_error))*100
    assert abs(score.mean()-r['target_score'])<1e-9
    def summarize(mask):
        n=int(mask.sum())
        return dict(rows=n,fraction=float(mask.mean()),PERM=float(score[mask].mean()) if n else None,
                    deficit_contribution=float(((100-score)*mask).mean()),
                    mean_log10_prediction_ratio=float(log_error[mask].mean()) if n else None)
    groups={
        'default_pred_default':actual_default&pred_default,
        'default_pred_continuous':actual_default&~pred_default,
        'continuous_pred_default':~actual_default&pred_default,
        'continuous_pred_continuous':~actual_default&~pred_default,
        'continuous_underprediction':~actual_default&(log_error<0),
        'continuous_overprediction':~actual_default&(log_error>0),
        'continuous_zero_utility':~actual_default&(np.abs(log_error)>=1),
    }
    result=dict(screening_fold=5,complete_five_folds=False,PERM=float(score.mean()),
                groups={name:summarize(mask) for name,mask in groups.items()},
                scope='Frozen descriptive diagnosis only. Overlapping sign/tail groups are not additive. No fitting, thresholds, row/well selection or input features derived from validation labels.')
    (STUDY/'error_source_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text='# PERM frozen error sources\n\n'+json.dumps(result,indent=2)+'\n'
    (STUDY/'ERROR_SOURCES.md').write_text(text,encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()

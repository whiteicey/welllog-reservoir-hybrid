"""Frozen final predictions only: descriptive paired RPS/likelihood gain attribution."""
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_ranked_probability_v151_20261001'

def main():
    d=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    ix=d['validation'];truth=d['truth'][ix];y=truth[:,0];wells=d['wells'][ix];default=base.dmask(truth)[:,0]
    point={};evidence={}
    for name,path in [('likelihood','pure_dl_por_joint_density_width_v140_fold5/wide256'),('joint_rps','pure_dl_por_ranked_probability_v151_fold5/joint_rps')]:
        out=ROOT/'outputs'/path;report=json.loads((out/'report.json').read_text(encoding='utf-8-sig'))
        assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
        p=np.load(out/'channel_pred.npy');assert p.shape==(144665,) and np.isfinite(p).all()
        point[name]=np.maximum(0,1-abs(p-y)/(.08*(abs(y)+1e-5)))*100
        assert abs(point[name].mean()-report['target_score'])<1e-9
        evidence[name]=dict(POR=report['target_score'],best_epoch=report['best_epoch'],source=str((out/'channel_pred.npy').relative_to(ROOT)))
    gain=point['joint_rps']-point['likelihood']
    perwell=[dict(well=int(w),rows=int((wells==w).sum()),delta=float(gain[wells==w].mean())) for w in np.unique(wells)]
    result=dict(screening_fold=5,complete_five_folds=False,rows=len(y),same_capacity_and_initialization=True,
        scope='Descriptive frozen paired predictions only; no fit, threshold tuning or row/well model choice',evidence=evidence,
        delta=float(gain.mean()),default_delta=float(gain[default].mean()),nondefault_delta=float(gain[~default].mean()),
        default_rows=int(default.sum()),nondefault_rows=int((~default).sum()),positive_wells=sum(v['delta']>0 for v in perwell),per_well=perwell)
    (STUDY/'gain_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=f"# Final paired jointCDF supervision analysis\n\nPOR delta {result['delta']:+.12f}; default {result['default_delta']:+.12f}, nondefault {result['nondefault_delta']:+.12f}; positive wells {result['positive_wells']}/16.\n\nSame architecture/capacity/initialization, only training loss differs. Single fixed fifth/seed; not fullOOF or independent generalization. True-label grouping is descriptive only, no label-derived inference inputs, row-wise choice or threshold tuning. Joint atom/continuous supervision does not guarantee improvement in default rows; inspect actual signs before attributing gain.\n"
    (STUDY/'GAIN_ANALYSIS.md').write_text(text,encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('per_well','evidence')}))

if __name__=='__main__':main()

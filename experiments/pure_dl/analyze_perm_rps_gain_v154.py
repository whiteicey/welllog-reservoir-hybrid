"""Final frozen PERM paired prediction diagnosis; no fitting or selection."""
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/perm_ranked_probability_v154_20261001'
def main():
    d=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    ix=d['validation'];truth=d['truth'][ix];y=truth[:,1];w=d['wells'][ix];default=base.dmask(truth)[:,1];points={};evidence={}
    for name,path in [('likelihood','pure_dl_perm_joint_density_v136_fold5/perm_conditional'),('joint_rps','pure_dl_perm_ranked_probability_v154_fold5/perm_conditional')]:
        out=ROOT/'outputs'/path;r=json.loads((out/'report.json').read_text(encoding='utf-8-sig'))
        assert r['predictions_reproduced_exactly'] and r['frozen_parameters_exact']
        p=np.load(out/'channel_pred.npy');assert p.shape==(144665,) and np.isfinite(p).all()
        points[name]=np.maximum(0,1-abs(np.log10(np.maximum(p/np.maximum(y,1e-5),1e-5))))*100
        assert abs(points[name].mean()-r['target_score'])<1e-9
        evidence[name]=dict(PERM=r['target_score'],best_epoch=r['best_epoch'],source=str((out/'channel_pred.npy').relative_to(ROOT)))
    gain=points['joint_rps']-points['likelihood'];perwell=[dict(well=int(v),rows=int((w==v).sum()),delta=float(gain[w==v].mean())) for v in np.unique(w)]
    result=dict(screening_fold=5,complete_five_folds=False,rows=len(y),same_capacity_and_initialization=True,evidence=evidence,
        delta=float(gain.mean()),default_delta=float(gain[default].mean()) if default.any() else None,
        nondefault_delta=float(gain[~default].mean()),positive_wells=sum(v['delta']>0 for v in perwell),per_well=perwell,
        scope='Descriptive frozen paired predictions only; no fit, label-derived inputs, thresholds or row/well selection')
    (STUDY/'gain_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (STUDY/'GAIN_ANALYSIS.md').write_text('# Final PERM jointCDF analysis\n\n'+json.dumps({k:v for k,v in result.items() if k not in ('per_well','evidence')},indent=2)+'\n\nSingle fifth/seed, not fullOOF or independent generalization.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_well','evidence')}))
if __name__=='__main__':main()

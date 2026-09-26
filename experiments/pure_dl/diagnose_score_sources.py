"""Read-only score decomposition; no fitting, oracle prediction, or model update."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import train_invariant as base
root=base.ROOT
spec=importlib.util.spec_from_file_location('official',root.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
folders=['pure_dl_lr55_cosine_1e3','pure_dl_continuous_difference_sequence']
results={}
for name in folders:
    folder=root/'outputs'/name
    assert json.loads((folder/'screening_verification.json').read_text())['passed']
    with np.load(folder/'oof.npz') as a:
        ix=a['fold']==0;truth=a['truth'][ix];pred=a['pred'][ix];well=a['well'][ix]
    with np.load(folder/'fold_1_distributions.npz') as a:gate=a['gate'].astype(float)
    defaults=base.dmask(truth);pd=base.dmask(pred)
    accuracy=np.maximum(0,1-np.abs(pred-truth)/(np.abs(truth)+1e-5)/np.array([.08,1,.05]))
    accuracy[:,1]=np.maximum(0,1-np.abs(np.log10(np.maximum(pred[:,1]/truth[:,1],1e-5))))
    weights=np.array([.3,.35,.35])
    total=float(100*(accuracy.mean(0)*weights).sum())
    assert abs(total-official.official_score(truth,pred)['TOTAL'])<1e-9
    targets={}
    for j,t in enumerate(base.TS):
        groups=[]
        for label,m in [('truth_default',defaults[:,j]),('truth_nondefault',~defaults[:,j]),('true_nondefault_pred_default',~defaults[:,j]&pd[:,j]),('true_nondefault_pred_nondefault',~defaults[:,j]&~pd[:,j]),('true_default_pred_nondefault',defaults[:,j]&~pd[:,j])]:
            groups.append({'group':label,'rows':int(m.sum()),'fraction':float(m.mean()),'score':float(accuracy[m,j].mean()*100) if m.any() else None,'weighted_total_points_lost':float(((1-accuracy[:,j])*m).mean()*weights[j]*100)})
        bins=[]
        edges=np.linspace(0,1,11)
        for k,lo in enumerate(edges[:-1]):
            m=(gate[:,j]>=lo)&((gate[:,j]<edges[k+1]) if k<9 else (gate[:,j]<=1))
            if m.any():bins.append({'lower':float(lo),'rows':int(m.sum()),'mean_probability':float(gate[m,j].mean()),'actual_default_rate':float(defaults[m,j].mean())})
        assert sum(b['rows'] for b in bins)==len(gate)
        targets[t]={'score':float(accuracy[:,j].mean()*100),'groups':groups,'gate_brier':float(((gate[:,j]-defaults[:,j])**2).mean()),'calibration_bins':bins}
    wells=[]
    for w in np.unique(well):
        m=well==w;s=official.official_score(truth[m],pred[m])
        wells.append({'well_index':int(w),'rows':int(m.sum()),'total':s['TOTAL'],'weighted_total_points_lost':float(m.mean()*(100-s['TOTAL']))})
    results[name]={'total':total,'targets':targets,'wells':wells}
first,second=[results[n] for n in folders]
deltas=[{'well_index':a['well_index'],'rows':a['rows'],'baseline':a['total'],'difference':b['total'],'delta':b['total']-a['total']} for a,b in zip(first['wells'],second['wells'])]
result={'scope':'fixed first fold descriptive analysis only; no fitted calibration/threshold/ensemble, no oracle, no new training score','models':results,'per_well_difference_deltas':deltas,'wells_improved':sum(x['delta']>0 for x in deltas),'wells_worsened':sum(x['delta']<0 for x in deltas)}
out=root/'outputs/score_source_diagnostic';out.mkdir(exist_ok=True)
(out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'baseline_targets':first['targets'],'largest_loss_wells':sorted(first['wells'],key=lambda x:-x['weighted_total_points_lost'])[:5],'difference_gain_wells':result['wells_improved'],'difference_loss_wells':result['wells_worsened']},indent=2))

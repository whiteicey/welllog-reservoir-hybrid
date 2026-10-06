"""CPU-only saved-prediction attribution; no fitting or rowwise prediction selection."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs'
SOURCE = OUT/'pure_dl_paired_local_context_router80_sequence'
STUDY = ROOT/'results/deep_learning/dual8222_round2_20260928'
ARMS = {
    'pointwise':'pure_dl_dual8222_heads_pointwise80_fold5',
    'local5':'pure_dl_dual8222_heads_local580_fold5',
    'linear':'pure_dl_dual8222_v2_linear80_fold5',
    'cross80':'pure_dl_dual8222_v2_cross8080_fold5',
    'gate_only':'pure_dl_dual8222_v3_gate_only80_fold5',
    'density_only':'pure_dl_dual8222_v3_density_only80_fold5',
}
TARGETS = ['POR','PERM','SW']
WEIGHTS = np.array([.30,.35,.35])


def utility(y, p):
    return np.column_stack([
        np.maximum(0,1-np.abs(p[:,0]-y[:,0])/(.08*(np.abs(y[:,0])+1e-5))),
        np.maximum(0,1-np.abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
        np.maximum(0,1-np.abs(p[:,2]-y[:,2])/(.05*(np.abs(y[:,2])+1e-5)))
    ])


def states(y):
    return np.column_stack([y[:,0]<=.1,y[:,1]<=.01,y[:,2]>=99.9])


def scores(values):
    s=values.mean(0)*100
    return dict(zip(TARGETS+['TOTAL'],list(s)+[float(s@WEIGHTS)]))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    with np.load(SOURCE/'oof.npz') as z:
        choose=z['fold']==4
        y=z['truth'][choose].astype(np.float64)
        wells=z['well'][choose]
    reference=np.load(SOURCE/'fold_5_pred.npy').astype(np.float64)
    assert y.shape==reference.shape==(144665,3)
    true_state=states(y); base_u=utility(y,reference)
    source_report=json.loads((SOURCE/'fold_5_report.json').read_text(encoding='utf-8'))
    assert abs(scores(base_u)['TOTAL']-source_report['scores']['TOTAL'])<1e-10
    result={'observed_at':datetime.now().astimezone().isoformat(),'source':str(SOURCE.relative_to(ROOT)),
            'source_scores':scores(base_u),'population':len(y),'runs':{},'pairwise':{},
            'limitations':'Original fifth is repeatedly used for development. No independent-test claim. Decompositions are descriptions, not label-based rules for inference.'}
    predictions={}
    for name,folder in ARMS.items():
        path=OUT/folder
        if not path.exists():continue
        status=json.loads((path/'runtime_status.json').read_text(encoding='utf-8')) if (path/'runtime_status.json').exists() else {}
        final=path/'fold_5_report.json'
        if not final.exists():
            result['runs'][name]={'status':status,'complete':False}
            continue
        report=json.loads(final.read_text(encoding='utf-8'))
        p=np.load(path/'fold_5_pred.npy').astype(np.float64)
        assert p.shape==y.shape and np.isfinite(p).all()
        u=utility(y,p); s=scores(u)
        assert all(abs(s[k]-report['scores'][k])<1e-9 for k in TARGETS+['TOTAL'])
        changes=(u-base_u)*100
        segments={}
        for c,t in enumerate(TARGETS):
            segments[t]={}
            for flag,label in [(True,'true_default'),(False,'true_nondefault')]:
                mask=true_state[:,c]==flag
                segments[t][label]={'n':int(mask.sum()),'source_score':float(base_u[mask,c].mean()*100),
                    'candidate_score':float(u[mask,c].mean()*100),
                    'contribution_to_TOTAL_delta':float((changes[:,c]*mask).mean()*WEIGHTS[c])}
        perwell=[]
        for well in np.unique(wells):
            mask=wells==well
            delta=changes[mask].mean(0)
            perwell.append({'well_index':int(well),'n':int(mask.sum()),'channel_delta':delta.tolist(),
                            'TOTAL_delta':float(delta@WEIGHTS),'TOTAL_contribution':float(delta@WEIGHTS*mask.mean())})
        detail={'complete':True,'folder':folder,'scores':s,'delta_source':s['TOTAL']-result['source_scores']['TOTAL'],
                'channel_deltas':dict(zip(TARGETS,changes.mean(0).tolist())),
                'true_label_population_decomposition':segments,'per_well':perwell,
                'well_wins':sum(x['TOTAL_delta']>1e-12 for x in perwell),
                'well_losses':sum(x['TOTAL_delta']< -1e-12 for x in perwell),
                'best_epochs':report['best_epochs'],'stopped_epochs':report['stopped_epochs'],
                'final_assembly_verified':report['assembled_heads_exactly_reproduce_selected_predictions'],
                'trainable_parameters':report['trainable_parameters'],'seconds':report['seconds']}
        assert abs(sum(v[label]['contribution_to_TOTAL_delta'] for v in segments.values() for label in v)-detail['delta_source'])<1e-9
        result['runs'][name]=detail
        predictions[name]=u
    for a,b in [('pointwise','linear'),('local5','linear'),('cross80','linear'),('gate_only','linear'),('density_only','linear')]:
        if a not in predictions or b not in predictions:continue
        difference=(predictions[a]-predictions[b])*100
        delta=difference.mean(0)
        wins=[float(difference[wells==w].mean(0)@WEIGHTS) for w in np.unique(wells)]
        result['pairwise'][a+'_minus_'+b]={'channel_delta':dict(zip(TARGETS,delta.tolist())),
            'TOTAL_delta':float(delta@WEIGHTS),'well_wins':sum(x>1e-12 for x in wins),
            'well_losses':sum(x< -1e-12 for x in wins)}
    dest=STUDY/'saved_prediction_attribution.json'
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(dest)
    print(json.dumps({**result,'runs':{k:{kk:vv for kk,vv in v.items() if kk!='per_well'} for k,v in result['runs'].items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

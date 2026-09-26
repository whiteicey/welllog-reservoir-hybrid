"""Fixed-weight exploratory V3+CNNMLP fusion, requiring verified V3 OOF provenance.

Never substitutes predictions from a full-80-well pretrained model for OOF.
Retains V3 default decisions; blends continuous values only when both models
predict nondefault. PERM is blended in log10 space. No outer-label tuning.
"""
import argparse,json
from pathlib import Path
import numpy as np
from train_cv import ROOT,scores,default_mask
from src.data import load_split,TARGET_COLUMNS


def blend(v3,nn,weight):
    assert 0<=weight<=1
    out=v3.copy();active=~default_mask(v3)&~default_mask(nn)
    for j in range(3):
        if j==1:mixed=10**((1-weight)*np.log10(np.maximum(v3[:,j],.01))+weight*np.log10(np.maximum(nn[:,j],.01)))
        else:mixed=(1-weight)*v3[:,j]+weight*nn[:,j]
        out[:,j]=np.where(active[:,j],mixed,v3[:,j])
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v3-oof',type=Path,required=True)
    ap.add_argument('--nn-oof',type=Path,default=ROOT/'outputs/cnn_mlp_cv/oof.npz')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/v3_nn_fusion')
    args=ap.parse_args()
    v=np.load(args.v3_oof,allow_pickle=False);n=np.load(args.nn_oof,allow_pickle=False)
    for key in ['truth','fold','well']:assert np.array_equal(v[key],n[key]),f'OOF alignment mismatch: {key}'
    assert v['pred'].shape==n['pred'].shape and np.isfinite(v['pred']).all()
    records=load_split(ROOT.parent/'data','train')
    assert np.array_equal(v['well_ids'],np.array([w for w,_ in records]))
    depths=np.concatenate([f.loc[f[TARGET_COLUMNS].notna().all(axis=1),'DEPTH'].to_numpy() for _,f in records])
    assert np.allclose(v['depth'],depths,atol=1e-6,rtol=0)
    ref=json.loads((ROOT/'results/assembly_D_report.json').read_text())
    baseline=scores(n['truth'],v['pred'])
    for t in TARGET_COLUMNS:assert abs(baseline[t]/100-ref[t]['nested']['micro'])<1e-6,f'V3 reference score mismatch: {t}'
    args.out.mkdir(parents=True,exist_ok=True)
    variants={}
    # Predeclared grid, report every result, no selection sold as unbiased evaluation.
    for w in [0.,.05,.1,.2,.3]:
        pred=blend(v['pred'],n['pred'],w)
        rep={'scores':scores(n['truth'],pred),'fold_scores':[],'well_wins':0}
        for f in range(5):
            ix=n['fold']==f;rep['fold_scores'].append(scores(n['truth'][ix],pred[ix]))
        for wi in np.unique(n['well']):
            ix=n['well']==wi
            rep['well_wins']+=int(scores(n['truth'][ix],pred[ix])['TOTAL']>scores(n['truth'][ix],v['pred'][ix])['TOTAL'])
        rep['delta_total']=rep['scores']['TOTAL']-baseline['TOTAL'];variants[str(w)]=rep
        np.save(args.out/f'prediction_weight_{w:.2f}.npy',pred)
    result={'v3':baseline,'cnn_mlp':scores(n['truth'],n['pred']),'variants':variants,
            'policy':'Fixed weights; retain V3 state decisions; blend only both-nondefault values, geometric for PERM.',
            'limitations':'Exploratory same-OOF comparison, not fresh independent validation. Final V3 OOF lacks latent gates/distributions; cannot test probabilistic gate fusion from this file alone.'}
    (args.out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

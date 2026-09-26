"""Verify all fold predictions against original train data and existing official metric."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split,TARGET_COLUMNS

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
o=np.load(a.out/'oof.npz');r=json.loads((a.out/'report.json').read_text())
rec=load_split(ROOT.parent/'data','train');truth=[];groups=[]
for i,(_,f) in enumerate(rec):
    valid=f[TARGET_COLUMNS].notna().all(1);truth.append(f.loc[valid,TARGET_COLUMNS].to_numpy());groups.append(np.full(valid.sum(),i))
y=np.concatenate(truth);g=np.concatenate(groups)
assert np.array_equal(y,o['truth']) and np.array_equal(g,o['well'])
fold=np.full(len(y),-1)
for f,(_,idx) in enumerate(GroupKFold(5).split(y,groups=g)):fold[idx]=f
assert np.array_equal(fold,o['fold'])
p=o['pred'];assert p.shape==(723556,3) and np.isfinite(p).all()
assert np.all(p>=[0,.01,8.305]) and np.all(p<=[50,10000,99.9])
spec=importlib.util.spec_from_file_location('official_metrics',ROOT.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
score=m.official_score(y,p)
assert abs(score['TOTAL']-r['oof_scores']['TOTAL'])<1e-8
for f in range(5):
    ix=fold==f;s=m.official_score(y[ix],p[ix]);assert abs(s['TOTAL']-r['folds'][f]['scores']['TOTAL'])<1e-8
    wid=[rec[i][0] for i in sorted(set(g[ix]))]
    assert wid==r['config']['outer_wells'][str(f)]
result={'passed':True,'rows':len(y),'wells':80,'score':score,'checks':['original labels unchanged','complete valid population','same GroupKFold','all five scores recomputed with existing metric','finite in bounds','reported validation well IDs match']}
(a.out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

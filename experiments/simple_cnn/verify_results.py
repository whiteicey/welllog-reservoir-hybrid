"""Independent acceptance: raw labels, well split, gating and existing official metric."""
import importlib.util
import json
from pathlib import Path
import sys
import argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split,TARGET_COLUMNS

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=ROOT/'outputs/simple_cnn_cv')
out=parser.parse_args().out
r=json.loads((out/'report.json').read_text())
a=np.load(out/'oof.npz')
records=load_split(ROOT.parent/'data','train')
y=[];groups=[]
for wi,(_,df) in enumerate(records):
    v=df[TARGET_COLUMNS].notna().all(axis=1)
    y.append(df.loc[v,TARGET_COLUMNS].to_numpy());groups.append(np.full(v.sum(),wi))
assert np.array_equal(np.concatenate(y),a['truth'])
assert np.array_equal(np.concatenate(groups),a['well'])
assert a['pred'].shape==(723556,3) and np.isfinite(a['pred']).all()
assert set(a['fold'])==set(range(5))
for wi,(wid,_) in enumerate(records):
    fold=np.unique(a['fold'][a['well']==wi]);assert len(fold)==1
    assert wid in r['config']['fold_wells'][str(int(fold[0]))]
assert sum(map(len,r['config']['fold_wells'].values()))==80
assert np.all((a['gate']>=0)&(a['gate']<=1))
assert np.array_equal(a['pred'],np.where(a['gate']>=.5,[.1,.01,99.9],a['continuous']))
assert np.all(a['pred']>=[0,.01,8.305]) and np.all(a['pred']<=[50,10000,99.9])
metric_path=ROOT.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py'
spec=importlib.util.spec_from_file_location('reference_metrics',metric_path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
official=m.official_score(a['truth'],a['pred'])
assert abs(official['TOTAL']-r['oof_scores']['TOTAL'])<1e-9
for f in range(5):
    mask=a['fold']==f
    score=m.official_score(a['truth'][mask],a['pred'][mask])
    assert abs(score['TOTAL']-r['folds'][f]['scores']['TOTAL'])<1e-9
result={'passed':True,'checks':['raw label alignment','80 wells each in one validation fold',
        'all OOF rows covered and finite','fixed 0.5 gate reproduction','bounds',
        'overall and all five folds match existing official metric'], 'official_scores':official}
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

"""Diagnose why fold 5 OOF (~80.x) lags other folds (82.5+)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TS = ['POR', 'PERM', 'SW']
DEPTHS = ['DEPTH']
CURVES = ['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']

def main(data_dir, out_json):
    files = sorted((Path(data_dir)/'train').glob('*.txt'))
    recs=[(p.stem, pd.read_csv(p, skiprows=[1], na_values=-99999)) for p in files]
    ys = [f[TS].to_numpy(np.float64) for _,f in recs]
    valid = [np.isfinite(v).all(1) for v in ys]
    truth = np.concatenate([v[m] for v,m in zip(ys,valid)])
    well = np.concatenate([np.full(m.sum(),i,np.int32) for i,m in enumerate(valid)])
    fold = np.full(len(truth),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(truth,groups=well)):
        fold[ix]=f

    well_fold = np.array([fold[np.flatnonzero(well==i)[0]] for i in range(80)])
    rows_per_fold = np.bincount(fold[fold>=0])

    diagnosis = {'rows_per_fold': rows_per_fold.tolist(), 'wells_per_fold': np.bincount(well_fold).tolist()}

    # Per-well stats
    well_stats = []
    starts = np.r_[0, np.cumsum([len(r[1]) for r in recs])]
    for i, (wn, frame) in enumerate(recs):
        # missing rates over valid rows only (both input curves + targets)
        missing = frame[CURVES].isna().mean().to_dict()
        depth_rng = float(frame['DEPTH'].max() - frame['DEPTH'].min())
        rows = int(valid[i].sum())
        # default-mask ratio among valid rows
        y = frame[TS].to_numpy(np.float64)
        dmask_ratios = {}
        for j, t in enumerate(TS):
            v = y[valid[i], j]
            if t == 'POR':
                thr = v <= 0.1
            elif t == 'PERM':
                thr = v <= 0.01
            else:
                thr = v >= 99.9
            dmask_ratios[t] = float(thr.mean())
        well_stats.append({
            'well': wn,
            'fold': int(well_fold[i]),
            'rows': rows,
            'depth_range': depth_rng,
            'curve_missing_mean': float(np.mean(list(missing.values()))),
            'curve_missing_max': float(np.max(list(missing.values()))),
            'default_ratio_POR': dmask_ratios['POR'],
            'default_ratio_PERM': dmask_ratios['PERM'],
            'default_ratio_SW': dmask_ratios['SW'],
        })
    diagnosis['wells'] = well_stats

    # Aggregate per fold
    agg = []
    for f in range(5):
        ws = [s for s in well_stats if s['fold'] == f]
        agg.append({
            'fold': f,
            'n_wells': len(ws),
            'rows_mean': float(np.mean([s['rows'] for s in ws])),
            'depth_range_mean': float(np.mean([s['depth_range'] for s in ws])),
            'missing_mean': float(np.mean([s['curve_missing_mean'] for s in ws])),
            'missing_max': float(np.max([s['curve_missing_max'] for s in ws])),
            'default_POR_mean': float(np.mean([s['default_ratio_POR'] for s in ws])),
            'default_PERM_mean': float(np.mean([s['default_ratio_PERM'] for s in ws])),
            'default_SW_mean': float(np.mean([s['default_ratio_SW'] for s in ws])),
            'default_SW_max': float(np.max([s['default_ratio_SW'] for s in ws])),
        })
    diagnosis['per_fold_summary'] = agg
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(agg, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'E:/Temp/welllog-repo/data',
         sys.argv[2] if len(sys.argv) > 2 else 'results/fold5_diagnosis.json')

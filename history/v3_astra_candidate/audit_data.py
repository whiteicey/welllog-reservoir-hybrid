#!/usr/bin/env python3
"""Read-only dataset/source/prediction audit; no training or leaderboard claims."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import INPUT_COLUMNS, TARGET_COLUMNS, read_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_dir
    report = {'wells': {}, 'source_comparison': {}, 'prediction_difference': {}}
    for split in ('train', 'test'):
        wells = []
        for p in sorted((root / 'data' / split).glob('*.txt')):
            raw = pd.read_csv(p, skiprows=[1])
            f = read_curve(p)
            info = {'well': p.stem, 'rows': len(f),
                    'depth_monotone': bool(f.DEPTH.is_monotonic_increasing),
                    'duplicate_depths': int(f.DEPTH.duplicated().sum()),
                    'median_step': float(f.DEPTH.diff().median()),
                    'absent_columns': [c for c in INPUT_COLUMNS if c not in f],
                    'curves': {c: {'missing_rate': float(f[c].isna().mean()),
                         'sentinel_rate': float((raw[c] == -99999).mean()),
                         'quantiles': [None if pd.isna(v) else float(v) for v in f[c].quantile([0, .05, .5, .95, 1])]}
                         for c in INPUT_COLUMNS if c in f}}
            if split == 'train':
                info['complete_target_rows'] = int(f[TARGET_COLUMNS].notna().all(axis=1).sum())
                info['targets'] = {}
                for t, d in [('POR', .1), ('PERM', .01), ('SW', 99.9)]:
                    y = f[t].dropna()
                    default = y >= d if t == 'SW' else y <= d
                    info['targets'][t] = {'valid_rows': len(y), 'default_rate': float(default.mean()),
                          'quantiles': y.quantile([0, .05, .25, .5, .75, .95, 1]).tolist()}
            wells.append(info)
        report['wells'][split] = wells
    for version in ('', '_v2'):
        with zipfile.ZipFile(root / f'submission{version}/submission_code.zip') as z:
            items = {}
            for name in z.namelist():
                if not name.endswith('.py'):
                    continue
                local = root / f'solution{version}' / name
                if version and name in ('train.py', 'predict.py'):
                    local = root / f'solution{version}' / name.replace('.py', '_v2.py')
                if local.is_file():
                    items[name] = {'same_bytes': hashlib.sha256(local.read_bytes()).digest() ==
                                                hashlib.sha256(z.read(name)).digest()}
            report['source_comparison'][version or 'v1'] = items
    old, new = [json.loads((root / p / 'result.json').read_text()) for p in ('submission', 'submission_v2')]
    old = {w['logId']: w['predictions'] for w in old['resultData']}
    for well in new['resultData']:
        a, b = old[well['logId']], well['predictions']
        if len(a) != len(b) or any(x['depth'] != y['depth'] for x, y in zip(a, b)):
            raise ValueError('Prediction depths do not align')
        report['prediction_difference'][well['logId']] = {
             t: {'mean_absolute_difference': float(np.mean(np.abs(np.array([r[t] for r in a]) - np.array([r[t] for r in b])))),
                 'changed_fraction': float(np.mean([x[t] != y[t] for x, y in zip(a, b)]))}
             for t in TARGET_COLUMNS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({s: {'wells': len(ws), 'rows': sum(w['rows'] for w in ws),
                     'min_length': min(w['rows'] for w in ws), 'max_length': max(w['rows'] for w in ws)}
                     for s, ws in report['wells'].items()}, indent=2))


if __name__ == '__main__':
    main()

"""Combine the three predeclared specialist columns without label-based selection."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from train import ROOT, TS, metric, write


def main():
    started = time.perf_counter()
    sources = [ROOT / 'outputs' / f'pure_dl_{t.lower()}_specialist' for t in TS]
    packs = [np.load(s / 'oof.npz') for s in sources]
    first = packs[0]
    for pack in packs[1:]:
        for field in ('truth', 'well', 'fold'):
            assert np.array_equal(pack[field], first[field]), field
    # Exactly one whole column from each corresponding specialist, fixed before
    # their validation scores were known. Labels are accessed only for scoring.
    pred = np.column_stack([pack['pred'][:, j] for j, pack in enumerate(packs)])
    assert pred.shape == (723556, 3) and np.isfinite(pred).all()
    out = ROOT / 'outputs/pure_dl_three_specialists'
    out.mkdir(parents=True, exist_ok=True)
    report0 = json.loads((sources[0] / 'report.json').read_text(encoding='utf-8'))
    config = {
        'method': 'predeclared full target columns; no row-dependent selection',
        'outer_wells': report0['config']['outer_wells'],
        'source_by_target': dict(zip(TS, map(str, sources))),
        'source_sha256': {t: hashlib.sha256((s / 'oof.npz').read_bytes()).hexdigest()
                          for t, s in zip(TS, sources)},
        'test_access': False,
    }
    y, fold, well = (first[k] for k in ('truth', 'fold', 'well'))
    np.savez_compressed(out / 'oof.npz', truth=y, pred=pred, fold=fold, well=well)
    folds = [{'fold': f+1, 'scores': metric(y[fold == f], pred[fold == f])} for f in range(5)]
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': folds,
              'seconds': time.perf_counter() - started}
    write(out / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

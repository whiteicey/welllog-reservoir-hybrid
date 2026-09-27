"""Verify overlap-sliding-window inference on fold-5 OOF and compare to non-overlap.

Reuses package_dl's Predictor but runs against the *training* wells that belong to
fold 5 (their labels exist), then scores with the official metric.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / 'package_dl'
sys.path.insert(0, str(PKG))
# Import the package_dl `src` package FIRST so it wins over the experiments/pure_dl
# namespace package that train_invariant installs on sys.path via ROOT/'package'.
import src.neural.paired_posterior_router_sequence  # noqa: E402,F401
sys.path.insert(0, str(REPO / 'experiments' / 'pure_dl'))
import train_invariant as base  # noqa: E402
from predict_heterogeneous_overlap import Predictor as OverlapPredictor  # noqa: E402
from predict_heterogeneous import Predictor as PlainPredictor  # noqa: E402


def main():
    data_dir = Path(sys.argv[1])
    model_root = Path(sys.argv[2])
    members = sys.argv[3].split(',')

    # Rebuild fold assignments exactly like training
    rec = base.load_split(data_dir, 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    truth = np.concatenate([v[m] for v, m in zip(ys, valid)])
    well = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    from sklearn.model_selection import GroupKFold
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=well)):
        fold[ix] = f

    # Fold-5 wells (FOLD_INDEX=4)
    FOLD_INDEX = 4
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    fold5_wells = [i for i in range(80) if fold[row_ids[i][0]] == FOLD_INDEX]
    print(f'fold5 wells: {fold5_wells}')

    # Run predictors on those wells
    plain = PlainPredictor(model_root, members.split(','), [1,2,3,4,5], 'cuda')
    overlap = OverlapPredictor(model_root, members.split(','), [1,2,3,4,5], 'cuda')

    pred_plain = np.full_like(truth, np.nan)
    pred_overlap = np.full_like(truth, np.nan)
    for wi in fold5_wells:
        _, frame = rec[wi]
        # map well-local valid rows to global rows
        valid_local = np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1)
        global_rows = row_ids[wi][valid_local]
        pred_plain[global_rows] = plain.predict_well(frame, batch_windows=32)[valid_local]
        pred_overlap[global_rows] = overlap.predict_well(frame, batch_windows=32)[valid_local]
        print(f'  well {wi}: {len(global_rows)} rows predicted')

    val_idx = np.flatnonzero(fold == FOLD_INDEX)
    score_plain = base.metric(truth[val_idx], pred_plain[val_idx])
    score_overlap = base.metric(truth[val_idx], pred_overlap[val_idx])
    print('PLAIN  fold5:', json.dumps(score_plain))
    print('OVERLAP fold5:', json.dumps(score_overlap))


if __name__ == '__main__':
    main()

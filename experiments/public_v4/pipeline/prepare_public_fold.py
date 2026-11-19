"""Prepare public fold caches for the original 18-stage v4 stage trainer."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'paper_workspace/v4_volve'))
from run_transfer_pdda2021 import load, TARGETS
from train_v4_public import PublicData

RUN = ROOT / 'paper_workspace/v4_volve/results/v4_root_20261113'
STAGES = RUN / 'stages'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def amplitude(frames, train_raw, out):
    curves = ('GR', 'PE', 'SP', 'CAL', 'AC', 'DEN', 'CNL', 'RXO', 'RT')
    med = {}
    for name in curves:
        vals = np.concatenate([frame[name].to_numpy(float) for _, frame in frames])
        finite = vals[np.isfinite(vals)]
        med[name] = float(np.median(finite)) if len(finite) else 0.0
    views = []
    for _, frame in frames:
        cols = []
        for name in curves:
            s = frame[name].replace([np.inf, -np.inf], np.nan).interpolate(method='linear', limit_direction='both').fillna(med[name])
            values = s.to_numpy(float)
            if name in ('GR', 'RXO', 'RT'):
                values = np.log1p(np.maximum(values, 0))
            s = pd.Series(values)
            cols.append(values)
            cols.extend(s.rolling(w, center=True, min_periods=1).mean().to_numpy() for w in (9, 51, 201))
            cols.extend(s.rolling(w, center=True, min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9, 51, 201))
        views.append(np.stack(cols, -1).astype(np.float32))
    x = np.concatenate(views)
    q = np.quantile(x[train_raw], [.25, .5, .75], axis=0)
    center = q[1].astype(np.float32)
    scale = np.maximum(q[2] - q[0], 1e-3).astype(np.float32)
    result = np.clip((x - center) / scale, -8, 8).astype(np.float16)
    np.savez_compressed(out, amplitude=result)
    return center, scale, med


def prepare(fold, test=False):
    tr_wells, te_wells = load()
    splits = list(GroupKFold(3).split(np.arange(len(tr_wells)), groups=np.arange(len(tr_wells))))
    fit, val = splits[fold - 1]
    fit_wells = [tr_wells[i] for i in fit]
    eval_wells = te_wells if test else [tr_wells[i] for i in val]
    frames = fit_wells + eval_wells
    data = PublicData(fit_wells, eval_wells, list(range(len(fit_wells))))
    root_ckpt = RUN / f'fold_{fold}.pt'
    if not root_ckpt.exists():
        raise FileNotFoundError(root_ckpt)
    master = torch.load(root_ckpt, map_location='cpu', weights_only=False)
    out = STAGES / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=True)
    if test:
        out = STAGES / f'fold_{fold}_test'
        out.mkdir(parents=True, exist_ok=True)
    fit_rows = data.train_mask_np & data.labeled_np
    truth = data.y[data.labeled_np]
    raw_to_valid = np.full(len(data.y), -1, dtype=np.int64)
    raw_to_valid[np.flatnonzero(data.labeled_np)] = np.arange(len(truth))
    train_raw = np.flatnonzero(fit_rows)
    validation = np.flatnonzero(data.well[data.labeled_np] != data.well[data.labeled_np])
    validation = np.flatnonzero(~data.train_mask_np[data.labeled_np])
    well = data.well[data.labeled_np]
    row_ids = [np.flatnonzero(well == w) for w, _ in frames]
    sample = np.stack([lo + np.minimum(((np.arange(128) + .5) * (hi - lo) / 128).astype(np.int64), hi - lo - 1)
                       for lo, hi in zip(data.starts[:-1], data.starts[1:])])
    bank = data.x[torch.as_tensor(sample, device='cuda')].detach()
    row_well = torch.as_tensor(np.repeat(np.arange(len(frames)), np.diff(data.starts)), device='cuda')
    fit_well_indices = list(range(len(fit_wells)))
    heldout_indices = list(range(len(fit_wells), len(frames)))
    reference = bank[torch.as_tensor(fit_well_indices, device='cuda')].mean(0)
    d = {
        'x': data.x, 'depth': data.depth, 'lower': data.lower, 'upper': data.upper,
        'absolute': data.absolute, 'labels': data.labels, 'default': data.default,
        'labeled': torch.from_numpy(fit_rows).cuda(), 'train_raw': torch.as_tensor(train_raw, device='cuda'),
        'truth': truth, 'fold': np.full(len(truth), fold - 1, dtype=np.int8), 'wells': well,
        'starts': data.starts, 'row_ids': row_ids, 'raw_to_valid': raw_to_valid,
        'validation': validation, 'contract': data.contract,
        'quality': torch.ones(len(data.x), device='cuda'), 'quality_ready': True,
        'qinfo': {'status': 'public_unit_weight', 'dynamic_updates': False},
        'population_rows': {'train_wells': fit_well_indices, 'heldout_wells': heldout_indices},
        'well_bank': bank, 'row_well': row_well,
    }
    metadata = {
        'centers': data.centers, 'density_edges': data.edges,
        'selected_feature_names': data.contract['whitelist'],
        'reference_inputs': reference,
        'bank_manifest': {'points_per_well': 128, 'features': 206, 'fit_wells': fit_well_indices,
                          'validation_fit': False, 'bank_labels_used': False},
        'sw_target_median': float(master.get('regression_median', master['target_medians'])[1]),
        'sw_target_scale': float(master.get('regression_scale', master['target_scales'])[1]),
    }
    amplitude_path = out / 'amplitude.npz'
    center, scale, med = amplitude(frames, train_raw, amplitude_path)
    torch.save({'data': d, 'metadata': metadata, 'root_sha256': sha(root_ckpt)}, out / 'data.pt')
    manifest = {
        'fold': fold, 'root_sha256': sha(root_ckpt), 'test_cache': test,
        'train_wells': [w for w, _ in fit_wells], 'evaluation_wells': [w for w, _ in eval_wells],
        'training_label_rows': len(train_raw), 'validation_label_rows': len(validation),
        'amplitude_medians': med, 'amplitude_center': center.tolist(), 'amplitude_scale': scale.tolist(),
        'labels_used_for_amplitude': False,
    }
    (out / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, choices=[1, 2, 3], required=True)
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    prepare(args.fold, args.test)

#!/usr/bin/env python3
"""Same-fold POR ablation. No parameter search on reported OOF labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from src.decision import por_prediction
from src.metrics import official_target_score
from train import make_por_bins, model_runtime, train_binary_classifier


def summarize(y, predictions, wells, folds):
    report = {}
    unique_wells = np.unique(wells)
    for name, pred in predictions.items():
        per_well = {str(w): official_target_score(y[wells == w], pred[wells == w], 'POR')
                    for w in unique_wells}
        per_fold = [official_target_score(y[folds == f], pred[folds == f], 'POR')
                    for f in np.unique(folds)]
        subsets = {'default': y <= .1, 'non_default': y > .1}
        for low, high in [(0.1, 1), (1, 5), (5, 10), (10, 20), (20, 50.0001)]:
            subsets[f'[{low},{high})_non_default'] = (y >= low) & (y < high) & (y > .1)
        report[name] = {
            'micro': official_target_score(y, pred, 'POR'),
            'macro': float(np.mean(list(per_well.values()))),
            'fold_scores': per_fold, 'fold_mean': float(np.mean(per_fold)),
            'fold_std': float(np.std(per_fold)), 'per_well': per_well,
            'worst_5_wells': sorted(per_well, key=per_well.get)[:5],
            'segments': {key: {'rows': int(mask.sum()), 'score':
                         official_target_score(y[mask], pred[mask], 'POR') if mask.any() else None}
                         for key, mask in subsets.items()},
        }
    # Resample whole wells, never rows. Paired interval is diagnostic, not nested CV.
    counts = np.array([(wells == w).sum() for w in unique_wells])
    rng = np.random.default_rng(941)
    draws = rng.integers(0, len(counts), (2000, len(counts)))
    baseline = report['argmax']['per_well']
    for name in predictions:
        if name == 'argmax':
            continue
        delta = np.array([report[name]['per_well'][str(w)] - baseline[str(w)] for w in unique_wells])
        boot = (delta[draws] * counts[draws]).sum(axis=1) / counts[draws].sum(axis=1)
        report[name]['paired_micro_delta_95pct'] = np.quantile(boot, [.025, .975]).tolist()
        report[name]['total_delta_if_other_targets_unchanged'] = 30 * (report[name]['micro'] - report['argmax']['micro'])
        report[name]['well_wins'] = int((delta > 0).sum())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='42,2026')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--binary-iterations', type=int, default=500)
    parser.add_argument('--ordinal-iterations', type=int, default=1200)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Use distinct seeds')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = load_split(args.data_dir, 'train')
    # Preserve v2 complete-case row population for fair within-run comparisons.
    valid = [f[TARGET_COLUMNS].notna().all(axis=1).to_numpy() for _, f in records]
    y = np.concatenate([f.loc[m, 'POR'].to_numpy(np.float64) for (_, f), m in zip(records, valid)])
    groups = np.concatenate([np.full(m.sum(), i, dtype=np.int32) for i, m in enumerate(valid)])
    well_names = np.array([w for w, _ in records])
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    predictions = {name: np.full(len(y), np.nan) for name in ('argmax', 'utility', 'joint')}
    manifest = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'gate': .5, 'scale': 1., 'offset': 0., 'splits': [],
                'note': 'Fresh fixed-gate POR comparison, NOT reproduction of historical tuned v2 score.',
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.rglob('*.py'))},
                'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((args.data_dir / 'train').glob('*.txt'))}}
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(args.folds).split(y, groups=groups)):
        fold_start = time.perf_counter()
        train_groups, val_groups = np.unique(groups[train_idx]), np.unique(groups[val_idx])
        assert not np.intersect1d(train_groups, val_groups).size
        builder = FeatureBuilder().fit([records[i][1] for i in train_groups])
        # Transform wells separately: rolling features never cross well boundaries.
        matrices = [builder.transform(f).base[m] for (_, f), m in zip(records, valid)]
        x = np.concatenate(matrices)
        del matrices
        non_default, classes, centers = make_por_bins(y[train_idx])
        probability = np.zeros((len(val_idx), len(centers)), dtype=np.float64)
        default_p = np.zeros(len(val_idx), dtype=np.float64)
        for seed in seeds:
            gate = train_binary_classifier(x[train_idx], y[train_idx] <= .1,
                                           args.binary_iterations, seed, args.cpu, 100)
            default_p += gate.predict_proba(x[val_idx])[:, 1] / len(seeds)
            model = CatBoostClassifier(iterations=args.ordinal_iterations, learning_rate=.04,
                    depth=8, loss_function='MultiClass', random_seed=seed + 100,
                    bootstrap_type='Bayesian', allow_writing_files=False, verbose=100,
                    **model_runtime(args.cpu))
            model.fit(Pool(x[train_idx][non_default], label=classes[non_default]))
            # Explicit class alignment also handles duplicate quantiles and absent classes.
            probability[:, model.classes_.astype(int)] += model.predict_proba(x[val_idx]) / len(seeds)
            del gate, model
        del x
        fold_ids[val_idx] = fold
        for mode in predictions:
            predictions[mode][val_idx] = por_prediction(probability, default_p, centers,
                {'threshold': .5, 'scale': 1., 'offset': 0.}, mode=mode)
        np.savez_compressed(args.output_dir / f'fold_{fold}.npz', indices=val_idx,
            probabilities=probability.astype(np.float32), default_probability=default_p,
            centers_log1p=centers, y=y[val_idx], wells=well_names[groups[val_idx]])
        manifest['splits'].append({'fold': fold, 'train_wells': well_names[train_groups].tolist(),
             'validation_wells': well_names[val_groups].tolist(),
             'seconds': time.perf_counter() - fold_start, 'medians': builder.medians_})
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    assert np.all(fold_ids >= 0) and all(np.isfinite(p).all() for p in predictions.values())
    summary = summarize(y, predictions, well_names[groups], fold_ids)
    summary['elapsed_seconds'] = time.perf_counter() - started
    summary['scope'] = 'POR only; implied total delta holds PERM and SW fixed; no tuning on OOF'
    np.savez_compressed(args.output_dir / 'oof.npz', y=y, wells=well_names[groups],
                        folds=fold_ids, **predictions)
    (args.output_dir / 'scores.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {m: v for m, v in d.items() if m not in ('per_well', 'segments')}
                      for k, d in summary.items() if isinstance(d, dict)}, indent=2))


if __name__ == '__main__':
    main()

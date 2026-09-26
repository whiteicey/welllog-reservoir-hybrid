"""Audit dense sequence CV: train-only rows/statistics and independent Gaussian CDF."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
import train_invariant as base

ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--folds', type=int, choices=range(1, 6), default=5)
args = ap.parse_args(); torch.set_num_threads(2)
cfg = json.loads((args.out/'config.json').read_text())
source = base.ROOT/'outputs/pure_dl_batchensemble_invariant'
assert json.loads((source/'verification.json').read_text())['passed']
assert json.loads((source/'interface_verification.json').read_text())['passed']
for relative, expected in cfg['dependency_sha256'].items():
    assert hashlib.sha256((base.ROOT/relative).read_bytes()).hexdigest() == expected
oof = np.load(args.out/'oof.npz'); fold = oof['fold']
rec = base.load_split(base.ROOT.parent/'data', 'train')
starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
mapping = np.concatenate([np.flatnonzero(np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1))+starts[i] for i, (_, frame) in enumerate(rec)])
assert len(mapping) == 723556
findings = []
for f in range(args.folds):
    ck = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    ref = torch.load(source/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg
    for key in ['feature_median', 'feature_scale', 'ple_edges_normalized', 'regression_median', 'regression_scale', 'centers', 'density_edges']:
        assert np.array_equal(np.asarray(ck[key]), np.asarray(ref[key])), (f, key)
    for key in ['medians', 'selected_feature_names', 'selected_feature_indices']:
        assert ck[key] == ref[key]
    assert np.array_equal(ck['training_raw_rows'], mapping[fold != f])
    assert np.array_equal(ck['validation_raw_rows'], mapping[fold == f])
    assert not np.intersect1d(ck['training_raw_rows'], ck['validation_raw_rows']).size
    assert cfg['private_heads'] == {'targets': 3, 'hidden_width': 64, 'initial_gain': .1, 'all_scales': True} and cfg['deep_supervision']
    state = ck['state_dict']
    assert state['main_head.first.weight'].shape == (192, 128, 1)
    assert state['main_head.second.weight'].shape == (384, 64, 1)
    assert state['stem.weight'].shape == (64, 208, 1)
    assert state['main_head.density.weight'].shape == (72, 128, 1)
    assert state['aux_coarse.density.weight'].shape == (72, 128, 1)
    assert cfg['epochs'] == 40 and cfg['steps'] == 64 and cfg['windows_per_step'] == 64
    assert cfg['seed'] == 20261006 and cfg['supervised_core'] == [64, 192]
    fr = json.loads((args.out/f'fold_{f+1}_report.json').read_text())
    assert fr['all_validation_rows_covered_once']
    errors = {}
    with np.load(args.out/f'fold_{f+1}_distributions.npz') as pack:
        assert np.array_equal(pack['indices'], np.flatnonzero(fold == f))
        gate = pack['gate']; assert gate.shape == (int((fold == f).sum()), 3)
        assert np.isfinite(gate).all() and ((gate >= 0) & (gate <= 1)).all()
        sample = np.linspace(0, len(gate)-1, 64, dtype=int)
        weight = pack['mixture_weight'][sample].astype(np.float64); weight /= weight.sum(-1, keepdims=True)
        mu = pack['mixture_mu'][sample].astype(np.float64); sd = pack['mixture_std'][sample].astype(np.float64)
        assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all()
        for j, name in enumerate(base.TS):
            p = pack[name+'_prob'].astype(np.float32)
            assert p.shape == (len(gate), 256) and np.isfinite(p).all() and (p >= 0).all()
            assert np.max(abs(p.sum(-1)-1)) < .001
            assert np.array_equal(pack[name+'_centers'], np.asarray(ck['centers'][j]))
            edges = np.asarray(ck['density_edges'][j])
            cdf = norm.cdf((edges[None, :, None]-mu[:, j, None, :])/sd[:, j, None, :])
            reconstructed = (np.diff(cdf, axis=1)*weight[:, j, None, :]).sum(-1)
            error = float(np.max(abs(reconstructed-p[sample])))
            assert error < .02, (f, name, error)
            errors[name] = error
    findings.append({'fold': f+1, 'same_audited_train_statistics': True,
                     'raw_loss_rows_exactly_training_fold': True, 'all_validation_rows_covered_once': True,
                     'independent_gaussian_cdf_error': errors})
result = {'passed': True, 'complete_five_folds': args.folds == 5, 'deep_supervision': cfg['deep_supervision'], 'folds': findings,
          'scope': 'Train-only statistics and raw-row mappings, architecture/budget, complete prediction coverage, probability validity and independent CDF reconstruction; original scoring verified separately'}
base.write(args.out/('private_head_sequence_verification.json' if args.folds == 5 else f'private_head_sequence_partial_{args.folds}.json'), result)
print(json.dumps(result, indent=2), flush=True)

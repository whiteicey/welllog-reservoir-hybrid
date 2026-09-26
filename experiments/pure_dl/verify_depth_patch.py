"""Validate depth-patch Transformer artifacts against the previously audited same-fold inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/pure_dl_depth_patch'
SOURCE = ROOT/'outputs/pure_dl_batchensemble_invariant'
cfg = json.loads((OUT/'config.json').read_text())
assert json.loads((SOURCE/'verification.json').read_text())['passed']
assert json.loads((SOURCE/'interface_verification.json').read_text())['passed']
for relative, expected in cfg['dependency_sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
findings = []
for fold in range(1, 6):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        a = torch.load(SOURCE/f'fold_{fold}.pt', map_location='cpu', weights_only=True)
        b = torch.load(OUT/f'fold_{fold}.pt', map_location='cpu', weights_only=True)
    for key in ['feature_median', 'feature_scale', 'ple_edges_normalized',
                'regression_median', 'regression_scale', 'centers', 'density_edges']:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), (fold, key)
    for key in ['medians', 'selected_feature_names', 'selected_feature_indices']:
        assert a[key] == b[key], (fold, key)
    for key in ['outer_wells', 'epochs', 'steps']:
        assert a['config'][key] == b['config'][key], (fold, key)
    with np.load(OUT/f'fold_{fold}_distributions.npz') as z:
        assert tuple(b['state_dict']['patch.weight'].shape) == (64, 44)
        assert tuple(b['state_dict']['position'].shape) == (17, 64)
        assert b['config']['batch_points'] == 1024 and b['config']['seed'] == 20261003
        gate = z['gate']
        assert np.isfinite(gate).all() and ((gate >= 0) & (gate <= 1)).all()
        ids = np.linspace(0, len(gate)-1, 64, dtype=int)
        weight = z['mixture_weight'][ids].astype(np.float64)
        weight /= weight.sum(-1, keepdims=True)
        mu = z['mixture_mu'][ids].astype(np.float64)
        scale = z['mixture_std'][ids].astype(np.float64)
        assert np.isfinite(scale).all() and (scale > 0).all()
        errors = {}
        for j, name in enumerate(['POR', 'PERM', 'SW']):
            p = z[name+'_prob'].astype(np.float32)
            assert p.shape == (len(gate), 256)
            assert np.isfinite(p).all() and (p >= 0).all()
            assert np.max(np.abs(p.sum(1)-1)) < .001
            edges = np.asarray(b['density_edges'][j])
            cdf = norm.cdf((edges[None, :, None]-mu[:, j, None, :])/scale[:, j, None, :])
            reconstructed = (np.diff(cdf, axis=1)*weight[:, j, None, :]).sum(-1)
            error = float(np.max(np.abs(reconstructed-p[ids])))
            # Stored components are float16; the export itself uses float32.
            assert error < .02, (fold, name, error)
            errors[name] = error
    findings.append({'fold': fold, 'same_audited_training_statistics': True,
                     'same_split_and_predeclared_budget': True,
                     'gaussian_cdf_reconstruction_max_abs_error': errors})
result = {'passed': True, 'folds': findings,
          'scope': 'Audited same-fold statistics, source SHA, probability validity and depth-patch architecture dimensions and independent Gaussian CDF reconstruction; official scores checked by verify.py'}
(OUT/'depth_patch_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

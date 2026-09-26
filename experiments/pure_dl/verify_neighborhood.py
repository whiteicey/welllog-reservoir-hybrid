"""CPU audit: full training-memory provenance and independent posterior samples."""
import hashlib
import argparse
import json
import numpy as np
import torch
from scipy.special import ndtr, softmax
from scipy.spatial.distance import cdist
import train_invariant as base
from neighborhood_density import NeighborhoodDensity

torch.set_num_threads(2)
parser = argparse.ArgumentParser()
parser.add_argument('--folds', type=int, default=5, choices=range(1, 6))
args = parser.parse_args()
OUT = base.ROOT/'outputs/pure_dl_neighborhood_density'
SOURCE = base.ROOT/'outputs/pure_dl_batchensemble_invariant'
cfg = json.loads((OUT/'config.json').read_text())
assert json.loads((SOURCE/'verification.json').read_text())['passed']
assert json.loads((SOURCE/'interface_verification.json').read_text())['passed']
for relative, sha in cfg['source_sha256'].items():
    assert hashlib.sha256((base.ROOT/relative).read_bytes()).hexdigest() == sha
artifact = np.load(OUT/'oof.npz')
y, fold, well = (artifact[k] for k in ('truth', 'fold', 'well'))
rec = base.load_split(base.ROOT.parent/'data', 'train')
default = base.dmask(y)
findings = []
for f in range(args.folds):
    # Checkpoints are local artifacts produced by the authorized runner.
    ck = torch.load(OUT/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    ref = torch.load(SOURCE/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    for key in ['feature_median', 'feature_scale', 'ple_edges_normalized', 'regression_median',
                'regression_scale', 'centers', 'density_edges']:
        assert np.array_equal(np.asarray(ck[key]), np.asarray(ref[key])), (f, key)
    for key in ['medians', 'selected_feature_names', 'selected_feature_indices']:
        assert ck[key] == ref[key], (f, key)
    assert ck['config'] == cfg
    training = np.flatnonzero(fold != f); validation = np.flatnonzero(fold == f)
    assert np.array_equal(training, ck['memory_rows'])
    assert np.array_equal(well[training], ck['memory_wells'])
    assert not set(well[training]) & set(well[validation])
    assert np.array_equal(default[training], ck['memory_default'].numpy())
    target_z = np.column_stack([(base.TRANSFORM[j](y[training, j])-ck['regression_median'][j])/ck['regression_scale'][j] for j in range(3)]).astype(np.float32)
    assert np.array_equal(target_z, ck['memory_target_z'].numpy())
    model = NeighborhoodDensity().eval(); model.load_state_dict(ck['state_dict'])
    builder = base.FeatureBuilder(); builder.medians_ = ck['medians']
    def embeddings(rows):
        result = np.empty((len(rows), 3, 32), np.float32)
        for wi in np.unique(well[rows]):
            frame = rec[int(wi)][1]
            _, enhanced = builder._transform_frames(frame)
            assert [enhanced.columns[i] for i in ck['selected_feature_indices']] == ck['selected_feature_names']
            feature = np.nan_to_num(enhanced.to_numpy(dtype=np.float32))[:, ck['selected_feature_indices']]
            valid = np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1)
            wellrows = np.flatnonzero(well == wi)
            mask = well[rows] == wi
            selected = feature[valid][np.searchsorted(wellrows, rows[mask])]
            # Match float32 preparation and the runner's stored float16 input table.
            selected = np.clip((selected-np.asarray(ck['feature_median'], np.float32))/np.asarray(ck['feature_scale'], np.float32), -8, 8).astype(np.float16).astype(np.float32)
            with torch.no_grad():
                result[mask] = model.encode(torch.from_numpy(selected)).numpy()
        return result
    sample = np.linspace(0, len(validation)-1, 4, dtype=int)
    query = embeddings(validation[sample])
    mem = ck['memory_embeddings'].numpy()
    mi = np.linspace(0, len(training)-1, 8, dtype=int)
    assert np.allclose(embeddings(training[mi]), mem[mi], atol=3e-5, rtol=1e-4)
    errors = {}
    with np.load(OUT/f'fold_{f+1}_distributions.npz') as pack:
        assert np.array_equal(pack['indices'], validation)
        gate = pack['gate']
        assert gate.shape == (len(validation), 3) and np.isfinite(gate).all()
        assert ((gate >= 0) & (gate <= 1)).all()
        for j, name in enumerate(base.TS):
            saved = pack[name+'_prob'].astype(np.float32)
            assert saved.shape == (len(validation), 256) and np.isfinite(saved).all()
            assert (saved >= 0).all() and np.max(abs(saved.sum(-1)-1)) < .001
            temp = float(model.log_temperature[j].exp().clamp(.05, 20).detach())
            weights = softmax(-cdist(query[:, j], mem[:, j])/temp, axis=1)
            reconstructed_gate = weights@default[training, j]
            assert np.max(abs(reconstructed_gate-gate[sample, j])) < .0001
            weights[:, default[training, j]] = 0
            weights /= weights.sum(-1, keepdims=True)
            sd = float(model.log_bandwidth[j].exp().clamp(.02, 1).detach())
            prob = np.zeros((len(sample), 256))
            for start in range(0, len(training), 4096):
                cdf = ndtr((np.asarray(ck['density_edges'][j])[None]-target_z[start:start+4096, j, None])/sd)
                kernel = np.diff(cdf, axis=1)
                kernel /= kernel.sum(-1, keepdims=True)
                prob += weights[:, start:start+4096]@kernel
            error = float(np.max(abs(prob-saved[sample])))
            assert error < .002, (f, name, error)
            errors[name] = error
    findings.append({'fold': f+1, 'all_memory_rows_are_outer_training': True,
                     'all_memory_labels_recomputed': True, 'same_audited_fold_statistics': True,
                     'sample_memory_embeddings_recomputed': True,
                     'independent_scipy_posterior_max_abs_error': errors})
    print('VERIFIED_FOLD', f+1, errors, flush=True)
result = {'passed': True, 'complete_five_folds': args.folds == 5, 'folds': findings, 'scope': 'Complete memory labels and provenance; audited training statistics; 4 validation posterior samples/fold independently reconstructed from full memory on CPU; official scoring checked separately'}
base.write(OUT/('neighborhood_verification.json' if args.folds == 5 else f'neighborhood_partial_{args.folds}.json'), result)
print(json.dumps(result, indent=2))

"""Audit direct-reward actions, training-only grids, and replay fixed witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
import train_invariant as base
from action_sequence_network import Network
from action_sequence_utils import make_actions, fit_normalizers

ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--folds', type=int, default=5, choices=range(1, 6))
args = ap.parse_args(); torch.set_num_threads(2)
cfg = json.loads((args.out/'config.json').read_text())
for relative, expected in cfg['dependency_sha256'].items():
    assert hashlib.sha256((base.ROOT/relative).read_bytes()).hexdigest() == expected
reference = base.ROOT/'outputs/pure_dl_multiscale_sequence'
assert json.loads((reference/'verification.json').read_text())['passed']
assert json.loads((reference/'sequence_verification.json').read_text())['passed']
oof = np.load(args.out/'oof.npz'); truth, fold = oof['truth'], oof['fold']
rec = base.load_split(base.ROOT.parent/'data', 'train')
starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
mapping = np.concatenate([np.flatnonzero(np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1))+starts[i] for i, (_, frame) in enumerate(rec)])
depth = np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _, frame in rec])
findings = []
for f in range(args.folds):
    ck = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    ref = torch.load(reference/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg
    for key in ['feature_median', 'feature_scale', 'ple_edges_normalized', 'regression_median', 'regression_scale']:
        assert np.array_equal(np.asarray(ck[key]), np.asarray(ref[key])), (f, key)
    for key in ['medians', 'selected_feature_names', 'selected_feature_indices']:
        assert ck[key] == ref[key]
    assert np.array_equal(ck['training_raw_rows'], mapping[fold != f])
    assert np.array_equal(ck['validation_raw_rows'], mapping[fold == f])
    actions = make_actions(truth[fold != f])
    for j in range(3):
        assert np.array_equal(actions[j], ck['actions'][j])
        assert np.array_equal(actions[j], cfg['action_grids'][str(f)][j])
    constants = fit_normalizers(torch.tensor(truth[fold != f], device='cuda', dtype=torch.float32),
                               [torch.tensor(a, device='cuda', dtype=torch.float32) for a in actions])
    assert np.allclose(constants, ck['loss_normalizers'], atol=1e-10, rtol=1e-10)
    model = Network([len(a) for a in actions]).cuda().eval(); model.load_state_dict(ck['state_dict'])
    assert sum(p.numel() for p in model.parameters()) == cfg['parameters_by_fold'][str(f)]
    assert cfg['seed'] == 20261006 and cfg['epochs'] == 40 and cfg['steps'] == 64
    assert cfg['windows_per_step'] == 64 and cfg['supervised_core'] == [64, 192]
    validation = np.flatnonzero(fold == f)
    with np.load(args.out/f'fold_{f+1}_actions.npz') as pack:
        assert np.array_equal(pack['indices'], validation)
        selected = pack['action_index']; probability = pack['action_probability']; margin = pack['action_margin']
        assert selected.shape == probability.shape == margin.shape == (len(validation), 3)
        assert np.isfinite(probability).all() and ((probability > 0) & (probability <= 1)).all()
        assert np.isfinite(margin).all() and (margin >= 0).all()
        wl = np.linspace(0, len(validation)-1, 64, dtype=int)
        assert np.array_equal(pack['witness_indices'], validation[wl])
        for j, name in enumerate(base.TS):
            assert np.array_equal(pack[name+'_actions'], actions[j])
            assert (selected[:, j] < len(actions[j])).all()
            assert np.array_equal(oof['pred'][validation, j], actions[j][selected[:, j]])
            saved = pack[name+'_witness_logits'].astype(np.float64)
            assert saved.shape == (64, len(actions[j])) and np.isfinite(saved).all()
            p = softmax(saved, axis=-1); winner = saved.argmax(-1)
            assert np.array_equal(winner, selected[wl, j])
            assert np.allclose(p[np.arange(64), winner], probability[wl, j], atol=1e-6)
            top = np.partition(p, -2, axis=-1)[:, -2:]
            assert np.allclose(top.max(-1)-top.min(-1), margin[wl, j], atol=1e-6)
        # Reconstruct every exported witness input from original INPUT curves and
        # this fold's stored training statistics, including padded well boundaries.
        builder = base.FeatureBuilder(); builder.medians_ = ck['medians']
        fm = np.asarray(ck['feature_median'], np.float32); fs = np.asarray(ck['feature_scale'], np.float32)
        feature_cache = {}
        for k in range(64):
            center, anchor = int(pack['witness_center'][k]), int(pack['witness_anchor'][k])
            wi = int(np.searchsorted(starts, anchor, side='right')-1)
            assert rec[wi][0] in cfg['outer_wells'][str(f)]
            index = center+np.arange(-128, 128); valid = (index >= starts[wi]) & (index < starts[wi+1])
            safe = np.clip(index, starts[wi], starts[wi+1]-1)
            assert np.array_equal(valid, pack['witness_valid'][k])
            position = int(pack['witness_position'][k]); assert valid[position]
            assert safe[position] == mapping[validation[wl[k]]]
            if wi not in feature_cache:
                _, enhanced = builder._transform_frames(rec[wi][1])
                assert [enhanced.columns[i] for i in ck['selected_feature_indices']] == ck['selected_feature_names']
                values = np.nan_to_num(enhanced.to_numpy(dtype=np.float32))[:, ck['selected_feature_indices']]
                feature_cache[wi] = np.clip((values-fm)/fs, -8, 8).astype(np.float16)
            assert np.array_equal(feature_cache[wi][safe-starts[wi]], pack['witness_features'][k])
            relative = np.tanh((depth[safe]-depth[anchor])/10)
            assert np.allclose(relative, pack['witness_relative_depth'][k], atol=5e-7, rtol=1e-6)
        errors = {name: 0. for name in base.TS}
        with torch.no_grad():
            for begin in (0, 32):
                x = torch.tensor(pack['witness_features'][begin:begin+32], device='cuda')
                d = torch.tensor(pack['witness_relative_depth'][begin:begin+32], device='cuda')
                v = torch.tensor(pack['witness_valid'][begin:begin+32], device='cuda')
                pos = torch.tensor(pack['witness_position'][begin:begin+32], device='cuda', dtype=torch.long)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(x, d, v)
                for j, name in enumerate(base.TS):
                    replay = output[j][torch.arange(32, device='cuda'), pos].float().softmax(-1).cpu().numpy()
                    expected = softmax(pack[name+'_witness_logits'][begin:begin+32].astype(np.float64), axis=-1)
                    errors[name] = max(errors[name], float(np.max(abs(replay-expected))))
        assert max(errors.values()) < .005, (f, 'witness replay', errors)
        findings.append({'fold': f+1, 'train_statistics_grids_and_constants_recomputed': True,
                         'all_prediction_actions_match_saved_indices': True,
                         'all64_witness_inputs_reconstructed_from_raw': True,
                         'all64_witness_decisions_recomputed': True,
                         'gpu_model_replay_probability_max_abs_error': errors})
    del model, feature_cache, ck, ref; torch.cuda.empty_cache()
    print('VERIFIED_FOLD', f+1, errors, flush=True)
result = {'passed': True, 'complete_five_folds': args.folds == 5, 'folds': findings,
          'scope': 'Original-input reconstruction, train-only grids/loss normalization, all action indices, independent witness argmax/softmax and GPU model replay; full official scores verified separately'}
base.write(args.out/('action_verification.json' if args.folds == 5 else f'action_partial_{args.folds}.json'), result)
print(json.dumps(result, indent=2), flush=True)

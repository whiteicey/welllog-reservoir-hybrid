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
ap.add_argument('--folds', type=int, choices=range(1, 6), default=1)
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
    assert cfg['trend_branch']['bins'] == 64 and cfg['trend_branch']['fit_steps'] == 600 and cfg['trend_branch']['frozen_before_residual_training']
    assert cfg['trend_branch']['depth_index'] == ck['selected_feature_names'].index('DEPTH_REL')
    assert cfg['screening_threshold'] == 82.5
    state = ck['state_dict']
    assert state['global_trend.layers.0.weight'].shape == (32,1)
    for key,value in ck['initial_global_trend_state'].items():
        assert torch.equal(state['global_trend.'+key],value)
    ix = cfg['trend_branch']['depth_index']
    assert np.isclose(float(state['depth_median']),ck['feature_median'][ix])
    assert np.isclose(float(state['depth_scale']),ck['feature_scale'][ix])
    assert state['stem.weight'].shape == (64, 208, 1)
    assert state['main_head.density.weight'].shape == (72, 128, 1)
    assert state['aux_coarse.density.weight'].shape == (72, 128, 1)
    # Rebuild the train-only trend profile from original depth and labels.
    from global_residual_sequence import average_profile
    us=[];zs=[];ds=[];ws=[]
    for wi,(_,frame) in enumerate(rec):
        mask=np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1)
        if not np.isin(starts[wi]+np.flatnonzero(mask)[0],ck['training_raw_rows']):continue
        depth_values=frame['DEPTH'].to_numpy(np.float32)
        u=(depth_values-depth_values.min())/max(float(depth_values.max())-float(depth_values.min()),1.)
        normalized=np.clip((u-np.float32(ck['feature_median'][ix]))/np.float32(ck['feature_scale'][ix]),-8,8).astype(np.float16)
        restored=np.clip(normalized.astype(np.float32)*np.float32(ck['feature_scale'][ix])+np.float32(ck['feature_median'][ix]),0,1)
        yy=frame[base.TS].to_numpy(np.float64)[mask]
        zz=np.column_stack([(base.TRANSFORM[j](yy[:,j])-ck['regression_median'][j])/ck['regression_scale'][j] for j in range(3)]).astype(np.float32)
        us.append(restored[mask]);zs.append(zz);ds.append(base.dmask(yy));ws.append(np.full(len(yy),wi))
    centers,profile,counts=average_profile(np.concatenate(us),np.concatenate(zs),np.concatenate(ds),np.concatenate(ws))
    fit=ck['global_trend_fit']
    assert np.array_equal(counts,np.asarray(fit['contributing_wells']))
    assert np.allclose(profile,np.asarray(fit['profile_z']),atol=1e-5,rtol=1e-6)
    assert np.array_equal(np.unique(np.concatenate(ws)),fit['well_ids']) and len(fit['well_ids'])==64
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
base.write(args.out/('global_residual_sequence_verification.json' if args.folds == 5 else f'global_residual_sequence_partial_{args.folds}.json'), result)
print(json.dumps(result, indent=2), flush=True)

# Reconstruct the complete original label population and fold assignment even for one-fold screening.
import importlib.util
from sklearn.model_selection import GroupKFold
truth = np.concatenate([frame.loc[np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1), base.TS].to_numpy(np.float64) for _,frame in rec])
well = np.concatenate([np.full(np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1).sum(), i) for i,(_,frame) in enumerate(rec)])
assert np.array_equal(truth,oof['truth']) and np.array_equal(well,oof['well'])
expected = np.full(len(truth),-1)
for f,(_,ix) in enumerate(GroupKFold(5).split(truth,groups=well)): expected[ix]=f
assert np.array_equal(expected,fold)
ix=fold==0; predictions=oof['pred'][ix]
assert np.isfinite(predictions).all() and (predictions >= [0,.01,8.305]).all() and (predictions <= [50,10000,99.9]).all()
spec=importlib.util.spec_from_file_location('official_metric',base.ROOT.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py')
metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)
score=metric_module.official_score(truth[ix],predictions)['TOTAL']
reported=json.loads((args.out/'screening_report.json').read_text())
assert abs(score-reported['fold1_scores']['TOTAL'])<1e-8
if args.folds==1:
    assert not reported['complete_five_folds']
    assert np.isnan(oof['pred'][~ix]).all() and not (args.out/'report.json').exists()
base.write(args.out/'screening_verification.json',{'passed':True,'first_fold_score':score,'qualifies_for_remaining_folds':score>82.5,'complete_five_folds':args.folds==5,'original_population_and_GroupKFold_verified':True})

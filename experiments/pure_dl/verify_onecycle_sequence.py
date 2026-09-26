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
    assert cfg['absolute_depth'] == 'train labeled rows median/IQR; clipped [-8,8]; zero initialized extra stem channel'
    original_depth = np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _,frame in rec]).astype(np.float64)
    depth_q = np.percentile(original_depth[mapping[fold != f]], [25,50,75])
    assert ck['absolute_depth_median'] == float(depth_q[1])
    assert ck['absolute_depth_scale'] == max(float(depth_q[2]-depth_q[0]), .1)
    assert cfg['normalization'] == 'LayerNorm across channels independently at each depth; nine blocks'
    assert cfg['screening_threshold'] == 82.5
    state = ck['state_dict']
    assert state['stem.weight'].shape == (64, 209, 1)
    assert state['main_head.density.weight'].shape == (72, 128, 1)
    assert state['aux_coarse.density.weight'].shape == (72, 128, 1)
    assert cfg['epochs'] == 40 and cfg['steps'] == 64 and cfg['windows_per_step'] == 64
    assert cfg['seed'] == 20261006 and cfg['supervised_core'] == [64, 192]
    fr = json.loads((args.out/f'fold_{f+1}_report.json').read_text())
    assert fr['all_validation_rows_covered_once']
    assert cfg['early_stopping'] == {'patience': 8, 'minimum_epochs': 10, 'maximum_epochs': 40, 'criterion': 'official_TOTAL', 'mode': 'max'}
    hist=json.loads((args.out/f'fold_{f+1}_validation_history.json').read_text())
    import math
    assert cfg['learning_rate_schedule'] == {'kind': 'OneCycleLR', 'total_steps': 2560, 'pct_start': .3, 'max_lr': .001, 'div_factor': 25., 'final_div_factor': 10000., 'cycle_momentum': True, 'base_momentum': .85, 'max_momentum': .95}
    def cosine(a,b,t):return b+(a-b)*.5*(1+math.cos(math.pi*t))
    for h in hist:
        for prefix,step in [('first',(h['epoch']-1)*64),('last',h['epoch']*64-1)]:
            if step<=767:
                expected_lr=cosine(.00004,.001,step/767);expected_beta=cosine(.95,.85,step/767)
            else:
                expected_lr=cosine(.001,.000000004,(step-767)/1792);expected_beta=cosine(.85,.95,(step-767)/1792)
            assert abs(h[prefix+'_applied_lr']-expected_lr)<1e-12
            assert abs(h[prefix+'_applied_beta1']-expected_beta)<1e-12
    values=[h['official_total'] for h in hist]
    best_index=int(np.argmax(values));assert ck['best_epoch']==best_index+1==fr['best_epoch']
    assert ck['stopped_epoch']==len(hist)==fr['stopped_epoch']
    assert abs(ck['best_validation_score']-max(values))<1e-10
    saved=torch.load(args.out/f'fold_{f+1}_best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==ck['best_epoch'] and saved['score']==ck['best_validation_score']
    assert all(torch.equal(value,saved['state_dict'][name]) for name,value in state.items())
    running=-float('inf');bad=0
    for index,h in enumerate(hist,1):
        if h['official_total']>running:running=h['official_total'];bad=0
        else:bad+=1
        assert h['bad_epochs']==bad and h['best_score']==running
        if index<len(hist):assert not (index>=10 and bad>=8)
    assert len(hist)==40 or (len(hist)>=10 and bad>=8)
    assert fr['best_checkpoint_predictions_reproduced']

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
base.write(args.out/('moe_head_sequence_verification.json' if args.folds == 5 else f'moe_head_sequence_partial_{args.folds}.json'), result)
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

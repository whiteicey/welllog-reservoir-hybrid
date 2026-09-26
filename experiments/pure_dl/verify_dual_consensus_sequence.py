"""Audit dense sequence CV: train-only rows/statistics and independent Gaussian CDF."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
import train_invariant as base
from fold5_screening_798 import FOLD_INDEX, THRESHOLD, fold_plan, qualifies

ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--folds', type=int, choices=[1,5], default=1)
args = ap.parse_args(); torch.set_num_threads(2)
cfg = json.loads((args.out/'config.json').read_text())
assert cfg['dropout_probability']==0.
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
for f in fold_plan(args.folds):
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
    assert cfg['screening_threshold'] == THRESHOLD and cfg['screening_fold']==5
    state = ck['state_dict']
    assert state['feature_expert.1.weight'].shape==(64,206)
    assert state['feature_expert.5.weight'].shape==(75,64)
    assert cfg['dual_evidence']['inference']=='sequence only'
    assert cfg['point_bypass'] is False
    assert cfg['absolute_depth_enabled'] is False
    assert cfg['residual_gate']=='constant0.5'
    expansions=[v for k,v in state.items() if k.endswith('expand.weight')]
    assert len(expansions)==9 and all(v.shape[0]==v.shape[1] for v in expansions)
    assert not any(k.startswith('point.') for k in state)
    assert state['combine.0.weight'].shape==(128,64,1)
    assert cfg['downsample_filter']=='fixed [.25,.5,.25] replicate padding before three stride2 convolutions'
    for name in ['down1','down2','down3']:
        assert torch.equal(state[name+'.kernel'],torch.tensor([.25,.5,.25]).reshape(1,1,3))
    assert state['stem.weight'].shape == (64, 208, 1)
    assert state['main_head.density.weight'].shape == (72, 128, 1)
    assert state['aux_coarse.density.weight'].shape == (72, 128, 1)
    assert cfg['loss_reduction'] == 'Global point weight times completed BCE plus masked NLL; original population denominator'
    assert cfg['epochs'] == 75 and cfg['steps'] == 64 and cfg['windows_per_step'] == 64
    assert cfg['seed'] == 20261006 and cfg['supervised_core'] == [64, 192]
    fr = json.loads((args.out/f'fold_{f+1}_report.json').read_text())
    assert fr['all_validation_rows_covered_once']
    assert cfg['early_stopping'] == {'patience': 18, 'minimum_epochs': 10, 'maximum_epochs': 75, 'criterion': 'official_TOTAL', 'mode': 'max'}
    hist=json.loads((args.out/f'fold_{f+1}_validation_history.json').read_text())
    import math
    assert cfg['initial_lr'] in [.0003,.001,.003] and cfg['lr_schedule'] in ['cosine','plateau']
    assert cfg['lr_policy']=={'minimum_lr':.00001,'plateau_factor':.5,'plateau_patience':3,'plateau_threshold':.01}
    lr=cfg['initial_lr'];plateau_best=-float('inf');plateau_bad=0
    for h in hist:
        if cfg['lr_schedule']=='cosine':
            for key,step in [('first_lr',(h['epoch']-1)*64),('last_lr',h['epoch']*64-1),('next_lr',h['epoch']*64)]:
                expected=.00001+(cfg['initial_lr']-.00001)*.5*(1+math.cos(math.pi*step/(75*64)))
                assert abs(h[key]-expected)<1e-11
        else:
            assert abs(h['first_lr']-lr)<1e-12 and abs(h['last_lr']-lr)<1e-12
            if h['official_total']>plateau_best+.01:plateau_best=h['official_total'];plateau_bad=0
            else:plateau_bad+=1
            if plateau_bad>3:lr=max(lr*.5,.00001);plateau_bad=0
            assert abs(h['next_lr']-lr)<1e-12
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
        if index<len(hist):assert not (index>=10 and bad>=18)
    assert len(hist)==75 or (len(hist)>=10 and bad>=18)
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
base.write(args.out/('moe_head_sequence_verification.json' if args.folds == 5 else 'fifth_fold_verification.json'), result)
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
ix=fold==FOLD_INDEX; predictions=oof['pred'][ix]
assert np.isfinite(predictions).all() and (predictions >= [0,.01,8.305]).all() and (predictions <= [50,10000,99.9]).all()
spec=importlib.util.spec_from_file_location('official_metric',base.ROOT.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py')
metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)
score=metric_module.official_score(truth[ix],predictions)['TOTAL']
reported=json.loads((args.out/'screening_report.json').read_text())
assert abs(score-reported['screening_scores']['TOTAL'])<1e-8
if args.folds==1:
    assert not reported['complete_five_folds']
    assert np.isnan(oof['pred'][~ix]).all() and not (args.out/'report.json').exists()
base.write(args.out/'screening_verification.json',{'passed':True,'screening_fold':5,'screening_fold_score':score,'qualifies_for_remaining_folds':qualifies(score),'complete_five_folds':args.folds==5,'original_population_and_GroupKFold_verified':True})

if args.folds==5:
    all_pred=oof['pred'];assert np.isfinite(all_pred).all()
    full=metric_module.official_score(truth,all_pred)
    report=json.loads((args.out/'report.json').read_text())
    assert abs(full['TOTAL']-report['oof_scores']['TOTAL'])<1e-8
    assert [r['fold'] for r in report['folds']]==[1,2,3,4,5]
    for f in range(5):
        mask=fold==f
        assert np.array_equal(all_pred[mask],np.load(args.out/f'fold_{f+1}_pred.npy'))
        assert abs(metric_module.official_score(truth[mask],all_pred[mask])['TOTAL']-report['folds'][f]['scores']['TOTAL'])<1e-8
    base.write(args.out/'full_oof_original_verification.json',{'passed':True,'complete_five_folds':True,'rows':len(truth),'wells':len(rec),'original_official_scores':full,'goal_over82':full['TOTAL']>82})

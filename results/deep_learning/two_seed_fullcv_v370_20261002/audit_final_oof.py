"""Once-only CPU audit of complete canonical OOF and uniform native fusion.

Does not train, select weights, rewrite FINAL, or establish unbiased accuracy.
Run after all five folds and FINAL exist; training provenance remains separate.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pure_dl'))
from fullcv_context_v97 import original_population, validate_metadata_source
from train_invariant import metric, TS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def check_scores(truth, prediction, expected):
    assert prediction.shape == truth.shape and np.isfinite(prediction).all()
    actual = metric(truth, prediction)
    assert all(abs(actual[key] - expected[key]) < 1e-9 for key in actual)
    return actual


def audit():
    destination = STUDY / 'FINAL_CANONICAL_OOF_AUDIT.json'
    assert not destination.exists(), 'Unchanged final OOF already audited'
    final = read(STUDY / 'FINAL.json')
    plan = read(STUDY / 'PLAN.json')
    population = original_population()
    n = len(population['truth'])
    assert n == 723556 and final['rows'] == n
    assert final['complete_original_five_fold_OOF'] and final['coverage_once']
    assert plan['fixed_weight_A'] == {'POR': .56, 'PERM': .44, 'SW': .68}
    assert plan['training_stages_per_model'] == 8
    assert final['stages_per_model'] == 8 and final['historical_stages_per_pair'] == 16
    with np.load(STUDY / 'OOF_POPULATION.npz') as saved:
        for key in ['truth', 'fold', 'wells', 'valid_to_raw']:
            assert np.array_equal(saved[key], population[key]), key
    labels = ['fixed', 'diagnostic', 'original', 'seed13']
    predictions = {key: np.load(STUDY / (key + '_OOF.npy')) for key in labels}
    scores = {key: check_scores(population['truth'], predictions[key], final['scores'][key])
              for key in labels}
    historical = ROOT / 'results/deep_learning/pairwise_complete_models_v368_20261002'
    fifth_plan = read(historical / 'AUGMENT_PLAN.json')
    assert fifth_plan['pairs'][0]['id'] == 'original__seed_20261013'
    fifth_models = {model['id']: model for model in fifth_plan['models']}
    historical_heads = {}
    historical_sources = {}
    fifth_source_paths = {
        'original': ROOT / 'outputs/pure_dl_paired_local_context_router80_sequence/fold_5.pt',
        'seed_20261013': ROOT / 'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261013/SOURCE/fold_5.pt',
    }
    for identity in ['original', 'seed_20261013']:
        model = fifth_models[identity]
        assert model['stage_count'] == 8
        source_path = fifth_source_paths[identity]
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        source_rows = validate_metadata_source(source, population, 4)
        assert source['config']['seed'] + 4 == (20261010 if identity == 'original' else 20261013)
        trace_path = source_path.with_name('fold_5_conflict_weights.npz')
        with np.load(trace_path) as trace:
            assert np.array_equal(np.sort(trace['raw_ids']), np.sort(source_rows['training_raw']))
            assert len(np.unique(trace['raw_ids'])) == len(trace['raw_ids'])
            prior = trace['epochs'][trace['epochs'] < source['best_epoch']]
            quality_epoch = int(prior[-1]) if len(prior) else 0
        historical_sources[identity] = dict(source_sha256=sha(source_path),
            original_training_validation_rows_exact=True, best_epoch=source['best_epoch'],
            quality_epoch=quality_epoch, trace_sha256=sha(trace_path))
        historical_heads[identity] = {}
        for target in TS:
            head = model['heads'][target]
            assert sha(Path(head['checkpoint'])) == head['sha256']
            cfg = read(Path(head['checkpoint']).with_name('config.json'))
            assert cfg['source_sha256'] == historical_sources[identity]['source_sha256']
            assert cfg['quality']['epoch'] == quality_epoch < source['best_epoch']
            assert cfg['quality']['sha256'] == historical_sources[identity]['trace_sha256']
            historical_heads[identity][target] = head['sha256']
        del source
    coverage = np.zeros(n, dtype=np.int8)
    folds = []
    assert [report['fold'] for report in final['folds']] == list(range(1, 6))
    for fold in range(1, 6):
        directory = STUDY / f'fold_{fold}'
        report = read(directory / 'FOLD_REPORT.json')
        assert report == final['folds'][fold - 1]
        validation = np.flatnonzero(population['fold'] == fold - 1)
        assert len(validation) == report['rows'] and (coverage[validation] == 0).all()
        coverage[validation] += 1
        train_wells = np.unique(population['wells'][population['fold'] != fold - 1])
        heldout_wells = np.unique(population['wells'][validation])
        assert len(train_wells) == 64 and len(heldout_wells) == 16
        assert not np.intersect1d(train_wells, heldout_wells).size
        assert report['fixed_weights'] == plan['fixed_weight_A']
        arrays = {key: np.load(directory / (key + '_prediction.npy')) for key in labels}
        for key in labels:
            assert np.array_equal(predictions[key][validation], arrays[key])
        check_scores(population['truth'][validation], arrays['fixed'], report['fixed_scores'])
        check_scores(population['truth'][validation], arrays['diagnostic'], report['diagnostic_scores'])
        for target_index, target in enumerate(TS):
            if fold == 5:
                parent = ROOT / 'results/deep_learning/pairwise_complete_models_v368_20261002'
                grid_path = parent / ('EXTRA_' + target + '_grids.npz')
                evidence = read(parent / ('EXTRA_' + target + '_GPU_FINAL.json'))
                assert evidence['native_disk_replay_exact'] and evidence['endpoints_exact']
            else:
                grid_path = directory / (target + '_grids.npz')
                evidence = read(directory / (target + '_GPU_REPORT.json'))
                assert all(evidence[key] for key in ['native_disk_replay_exact', 'endpoints_exact', 'coverage_once'])
            with np.load(grid_path) as grid:
                actions, weights = grid['actions'], grid['weights']
                indices = grid['indices'][0] if fold == 5 else grid['indices']
                assert indices.shape == (len(validation), 101)
                assert np.allclose(weights, np.arange(101) / 100, rtol=0, atol=1e-12)
                if fold != 5:
                    assert np.array_equal(grid['validation'], validation)
                fixed_index = int(round(plan['fixed_weight_A'][target] * 100))
                selected = report['per_head_best_weights'][target]
                diagnostic_index = int(round(selected['weight_original'] * 100))
                assert 0 <= diagnostic_index <= 100
                assert abs(weights[diagnostic_index] - selected['weight_original']) < 1e-12
                for key, index in [('fixed', fixed_index), ('diagnostic', diagnostic_index),
                                   ('original', 100), ('seed13', 0)]:
                    assert np.array_equal(arrays[key][:, target_index], actions[indices[:, index]])
        folds.append(dict(fold=fold, rows=len(validation), fixed_TOTAL=report['fixed_scores']['TOTAL'],
                          uniform_native_grid_columns_exact=True, train_wells=64, heldout_wells=16))
    assert (coverage == 1).all()
    code_hashes = {name: sha(ROOT / name) == digest
                   for name, digest in read(STUDY / 'PREPARATION.json')['code_sha256'].items()}
    assert all(code_hashes.values())
    assert final['fixed_strictly_above82'] == (scores['fixed']['TOTAL'] > 82)
    result = dict(observed_epoch=time.time(), rows=n, folds=folds, scores=scores,
                  canonical_population_and_group_split_exact=True, coverage_once=True,
                  unchanged_uniform_fixed_weights=plan['fixed_weight_A'],
                  OOF_exact_to_fold_files=True, frozen_code_hashes=code_hashes,
                  reused_fifth_head_checkpoint_sha256=historical_heads,
                  reused_fifth_source_row_and_Q_provenance=historical_sources,
                  artifact_sha256={key: sha(STUDY / (key + '_OOF.npy')) for key in labels},
                  fixed_strictly_above82=scores['fixed']['TOTAL'] > 82, CPU_only=True,
                  scope='Canonical OOF assembly and native uniform fusion audit; training provenance is separate',
                  fixed_scope='Pair and ratios selected on fifth; selection boundary remains disclosed',
                  diagnostic_scope='Each scored fold selects its head ratios; tuning diagnostic, not unbiased accuracy')
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(rows=n, scores=scores, fixed_strictly_above82=result['fixed_strictly_above82'])))


if __name__ == '__main__':
    audit()

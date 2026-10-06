"""Final CPU audit of actual remaining-fold checkpoints and load provenance.

Run once after the canonical final OOF audit. Historical fifth-fold provenance
is deliberately separate; this does not claim an unbiased validation estimate.
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
from fullcv_pair_context_v370 import STAGES, BASE_SEEDS, own_dir
from fullcv_context_v97 import original_population, validate_metadata_source


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


HASHES = {}


def sha(path):
    path = Path(path).resolve()
    if path not in HASHES:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(part)
        HASHES[path] = digest.hexdigest()
    return HASHES[path]


def audit():
    destination = STUDY / 'FINAL_REMAINING_TRAINING_AUDIT.json'
    assert not destination.exists(), 'Unchanged training evidence already audited'
    canonical = read(STUDY / 'FINAL_CANONICAL_OOF_AUDIT.json')
    assert canonical['canonical_population_and_group_split_exact'] and canonical['coverage_once']
    population = original_population()
    parents = {'POR113': 'SOURCE', 'POR151': 'POR113', 'PERM154': 'SOURCE',
               'SW104': 'SOURCE', 'POR341': 'POR151', 'PERM336': 'PERM154', 'SW351': 'SW104'}
    models = []
    for fold in range(1, 5):
        validation = np.flatnonzero(population['fold'] == fold - 1)
        training = np.flatnonzero(population['fold'] != fold - 1)
        for identity, base_seed in BASE_SEEDS.items():
            own = own_dir(identity, fold).resolve()
            assembled = read(own / 'COMPLETE_MODEL.json')
            assert assembled['stages'] == list(STAGES) and assembled['training_stages'] == 8
            assert assembled['seed'] == base_seed + fold - 1
            source = own / 'SOURCE' / f'fold_{fold}.pt'
            master = torch.load(source, map_location='cpu', weights_only=False)
            validate_metadata_source(master, population, fold - 1)
            assert np.array_equal(np.sort(master['training_raw_rows']),
                                  np.sort(population['valid_to_raw'][training]))
            assert np.array_equal(master['validation_raw_rows'], population['valid_to_raw'][validation])
            descriptor = torch.load(own / 'SOURCE/prepared_data.pt', map_location='cpu', weights_only=False)
            shared = own.parent / 'shared_input_data.pt'
            assert Path(descriptor['shared_input_path']).resolve() == shared
            assert descriptor['shared_input_sha256'] == sha(shared)
            assert descriptor['source_sha256'] == sha(source)
            assert descriptor['qinfo']['epoch'] < master['best_epoch']
            source_cfg = read(own / 'SOURCE/config.json')
            assert source_cfg == master['config']
            assert source_cfg['seed'] == base_seed
            assert source_cfg['original_fold'] == fold and source_cfg['model_identity'] == identity
            assert source_cfg['epochs'] == source_cfg['early_stopping']['maximum_epochs'] == 80
            assert source_cfg['early_stopping']['patience'] == 10
            assert source_cfg['early_stopping']['patience_after_strict_SOTA'] == 14
            assert source_cfg['features'] == 206 and source_cfg['test_access'] is False
            assert len(master['selected_feature_names']) == 206
            assert not any(any(part in name for part in ('DEVI', 'AZIM', 'BIT', 'CASE'))
                           for name in master['selected_feature_names'])
            stages = []
            for stage in STAGES:
                checkpoint = source if stage == 'SOURCE' else own / stage / 'model_best.pt'
                if stage == 'SOURCE':
                    # The original first SOURCE was attached before terminal snapshots
                    # were introduced. Its actual report/timing are authoritative.
                    terminal = dict(report=read(own / 'SOURCE' / f'fold_{fold}_report.json'),
                                    timing=read(own / 'SOURCE/SOURCE_TIMING.json'),
                                    checkpoint_sha256=sha(source))
                else:
                    terminal = read(STUDY / f'fold_{fold}' / (identity + '_' + stage + '_TERMINAL.json'))
                assert terminal['checkpoint_sha256'] == assembled['checkpoint_sha256'][stage] == sha(checkpoint)
                assert terminal['timing']['terminal'] and terminal['timing']['exit_code'] in (0, None)
                cfg = source_cfg if stage == 'SOURCE' else read(own / stage / 'config.json')
                hashes = cfg['dependency_sha256'] if stage == 'SOURCE' else cfg['source_hashes']
                assert all(sha(ROOT / name) == value for name, value in hashes.items())
                report = terminal['report']
                assert report == assembled['reports'][stage]
                assert 10 <= report['stopped_epoch'] <= 80
                assert 0 <= report['best_epoch'] <= report['stopped_epoch']
                if stage != 'SOURCE':
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    assert saved['config'] == cfg and saved['best_epoch'] == report['best_epoch']
                    assert saved['target_score'] == report['target_score']
                    assert cfg['original_fold'] == fold and cfg['model_identity'] == identity
                    assert cfg['seed'] == base_seed + fold - 1
                    assert cfg['epochs'] == 80 and cfg['patience'] == 10
                    assert cfg['patience_after_strict_SOTA'] == 14 and cfg['cosine_T_max'] == 80 * 64
                    assert cfg['total_recipe_training_stages'] == 8 and cfg['own_target_labels_only']
                    assert cfg['quality'] == descriptor['qinfo'] and cfg['source_sha256'] == sha(source)
                    parent = source if parents[stage] == 'SOURCE' else own / parents[stage] / 'model_best.pt'
                    assert cfg['warm_parent_sha256'] == sha(parent)
                    loads = cfg['actual_checkpoint_loads']
                    assert loads
                    for item in loads:
                        path = Path(item['path']).resolve()
                        assert path.is_relative_to(own) or path == shared, path
                        assert item['sha256'] == sha(path), path
                    assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
                    del saved
                else:
                    assert report['best_checkpoint_predictions_reproduced']
                stages.append(dict(stage=stage, best_epoch=report['best_epoch'],
                                   stopped_epoch=report['stopped_epoch'], checkpoint_sha256=sha(checkpoint)))
            models.append(dict(identity=identity, fold=fold, actual_seed=base_seed + fold - 1,
                               training_rows=len(training), validation_rows=len(validation),
                               own_Q=descriptor['qinfo'], stages=stages))
            del master, descriptor
    result = dict(observed_epoch=time.time(), models=models, remaining_fold_models=8,
                  stages_per_model=8, actual_checkpoint_loads_scoped_and_exact=True,
                  original_training_validation_raw_rows_exact=True, own_source_Q_before_best=True,
                  dependency_source_hashes_exact=True, CPU_only=True,
                  scope='Folds1-4 actual checkpoints/configs/load traces and source row provenance. '
                        'Historical fifth and unbiased accuracy are separate requirements.')
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(remaining_fold_models=len(models), stages_per_model=8, CPU_only=True)))


if __name__ == '__main__':
    audit()

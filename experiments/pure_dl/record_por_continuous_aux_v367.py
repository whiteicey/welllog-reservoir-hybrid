"""Terminal CPU assembly for held V367; no training, GPU, or fold launch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import psutil
import torch

from train import dmask
from train_invariant import metric, TS
from train_dual8222_heads80 import atomic_json, atomic_array
from fold5_screening_808 import qualifies

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'results/deep_learning/por_continuous_aux_v367_20261002'
OUT = ROOT / 'outputs/pure_dl_por_continuous_aux_v367_fold5'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_trainer_live(state):
    try:
        process = psutil.Process(state.get('pid', -1))
        return any(Path(arg).name == 'train_por_continuous_aux_v367.py' for arg in process.cmdline())
    except (psutil.Error, ValueError):
        return False


def assemble(stage):
    plan = read(STUDY / 'PLAN.json')
    phase = OUT / stage
    state_file = phase / 'runtime_status.json'
    if not state_file.exists():
        return dict(stage=stage, status='not_started', unscored=True)
    state = read(state_file)
    if own_trainer_live(state) or state.get('status') not in ('complete', 'complete_early_terminated'):
        return dict(stage=stage, status=state.get('status'), live=own_trainer_live(state), unscored=True)
    report = read(phase / 'report.json')
    config = read(phase / 'config.json')
    assert report['target'] == 'POR' and report['predictions_reproduced_exactly']
    assert report['frozen_parameters_exact'] and report['validation_rows'] == 144665
    assert config['trainable_parameters'] == plan['trainable_parameters']
    assert config['continuous_auxiliary_coefficient'] == plan['loss_coefficients'][stage]
    matched_configuration = json.loads(json.dumps(config))
    matched_configuration.pop('stage')
    matched_configuration.pop('continuous_auxiliary_coefficient')
    matched_configuration['initialization'].pop('continuous_coefficient')
    result_file = STUDY / f'{stage}_assembled.json'
    checkpoint_hash = sha(phase / 'model_best.pt')
    prediction_hash = sha(phase / 'channel_pred.npy')
    if result_file.exists():
        cached = read(result_file)
        assert cached['checkpoint_sha256'] == checkpoint_hash
        assert cached['POR_prediction_sha256'] == prediction_hash
        return cached

    # Match the original eight-stage fixed336/351, never the old six-stage154/104.
    fixed_paths = {}
    expected = [report['target_score']]
    for target in ('PERM', 'SW'):
        item = plan['fixed_other_heads'][target]
        checkpoint = Path(item['checkpoint'])
        assert sha(checkpoint) == item['sha256'], f'Fixed {target} checkpoint changed'
        fixed_paths[target] = checkpoint.parent / 'channel_pred.npy'
        fixed_report = read(checkpoint.parent / 'report.json')
        assert fixed_report['target'] == target and fixed_report['predictions_reproduced_exactly']
        expected.append(fixed_report['target_score'])
    data = torch.load(
        ROOT / 'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        map_location='cpu', weights_only=False,
    )
    truth = np.asarray(data['truth'][data['validation']])
    del data
    prediction = np.column_stack((
        np.load(phase / 'channel_pred.npy'),
        np.load(fixed_paths['PERM']), np.load(fixed_paths['SW']),
    ))
    assert prediction.shape == truth.shape == (144665, 3) and np.isfinite(prediction).all()
    scores = metric(truth, prediction)
    assert all(abs(scores[target] - value) < 1e-9 for target, value in zip(TS, expected))
    defaults = dmask(truth)
    strata = {}
    for column, target in enumerate(TS):
        strata[target] = {}
        for kind, mask in [('DEFAULT', defaults[:, column]), ('CONT', ~defaults[:, column])]:
            strata[target][kind] = dict(rows=int(mask.sum()), score=metric(truth[mask], prediction[mask])[target])
    result = dict(
        stage=stage, scores=scores, strata=strata, report=report,
        screening_fold=5, validation_rows=144665,
        total_training_stages=8, source_stage_counted=True,
        added_head_stages=dict(POR=1, PERM=1, SW=1),
        fixed_other_heads=plan['fixed_other_heads'],
        matched_configuration=matched_configuration,
        fixed_other_prediction_sha256={target: sha(path) for target, path in fixed_paths.items()},
        checkpoint_sha256=checkpoint_hash, POR_prediction_sha256=prediction_hash,
        gain_vs_original_POR=scores['POR'] - plan['SOTA_target_reference'],
        gain_vs_original_TOTAL=scores['TOTAL'] - plan['SOTA_TOTAL_reference'],
        strict_808_passed=qualifies(scores['TOTAL']),
        clear_gain_requires_matched_pair_interpretation=True,
        complete_five_folds=False, remaining_folds_started=False,
        CPU_saved_prediction_readback=True, independent_model_disk_reload=False,
        auxiliary_used_in_inference=False,
        total_recipe_wall_seconds=None,
        timing_scope='Report seconds cover one added phase only; no complete-recipe timing measured',
    )
    atomic_array(STUDY / f'{stage}_assembled_pred.npy', prediction)
    atomic_json(result_file, result)
    note = (
        f'\nV367 {stage} terminal CPU saved-prediction assembly: {json.dumps(scores)}, '
        f'8 honest stages/POR one addition, best{report["best_epoch"]}/stop{report["stopped_epoch"]}, '
        f'strict808={result["strict_808_passed"]}, TOTALdelta{result["gain_vs_original_TOTAL"]:+.12f}. '
        'Fixed PERM336/SW351, all144665 checked; no independent model disk reload, '
        'other folds or complete-recipe speed claim. Matched auxiliary gain awaits both candidates.\n'
    )
    for path in (STUDY / 'README.md', STUDY / 'PROJECTS.md', ROOT / 'PROJECTS.md', ROOT.parent / 'PROJECTS.md'):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(note)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('aux0', 'aux01'), required=True)
    stage = parser.parse_args().stage
    result = assemble(stage)
    paired = [STUDY / f'{name}_assembled.json' for name in ('aux0', 'aux01')]
    if all(path.exists() for path in paired) and not (STUDY / 'MATCHED_COMPARISON.json').exists():
        control, candidate = map(read, paired)
        assert control['fixed_other_heads'] == candidate['fixed_other_heads']
        assert control['matched_configuration'] == candidate['matched_configuration'], 'Pair differs beyond auxiliary coefficient'
        comparison = dict(
            control=control['scores'], candidate=candidate['scores'],
            same_capacity_auxiliary_gain={key: candidate['scores'][key] - control['scores'][key] for key in (*TS, 'TOTAL')},
            POR_strata_gain={kind: candidate['strata']['POR'][kind]['score'] - control['strata']['POR'][kind]['score'] for kind in ('DEFAULT', 'CONT')},
            candidate_vs_historical_TOTAL=candidate['gain_vs_original_TOTAL'],
            screening_fold=5, complete_five_folds=False, remaining_folds_started=False,
            no_automatic_adoption_or_fold_launch=True,
        )
        atomic_json(STUDY / 'MATCHED_COMPARISON.json', comparison)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

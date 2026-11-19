"""Run the complete public 18-stage v4 chain and evaluate its three final heads."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path('<PROJECT_ROOT>')
HERE = ROOT / 'paper_workspace/v4_volve/full_runtime'
RUN = ROOT / 'paper_workspace/v4_volve/results/v4_root_20261113'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from fullcv_models_v97 import STAGES, fold_dir, stage_dir
from train_invariant import metric as public_metric
from run_transfer_pdda2021 import rmse3, rmse_per, tol_acc

# Public output order is PHIF, SW, VSH (not the internal POR, PERM, SW order).
FINAL_HEADS = ('por_anchored', 'sw_density', 'perm_split')
STATUS = RUN / 'stages/controller_status.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_status(**kwargs):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({'pid': os.getpid(), 'updated_at': time.time(), **kwargs}, ensure_ascii=False, indent=2), encoding='utf-8')


def run_script(name, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-u', str(HERE / name), *map(str, args)], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'{name} failed with {result.returncode}; see {log}')


def main():
    started = time.perf_counter()
    write_status(status='running', current='initializing', completed=[])
    for f in (1, 2, 3):
        if not (fold_dir(f) / 'data.pt').exists():
            run_script('prepare_public_fold.py', ['--fold', f], fold_dir(f) / 'prepare.log')
        if not (fold_dir(f).with_name(f'fold_{f}_test') / 'data.pt').exists():
            run_script('prepare_public_fold.py', ['--fold', f, '--test'], fold_dir(f).with_name(f'fold_{f}_test') / 'prepare.log')
        for stage, *_ in STAGES:
            out = stage_dir(f, stage)
            if (out / 'report.json').exists():
                continue
            if stage == 'por_mean' and not (fold_dir(f) / 'selected_factors.pt').exists():
                run_script('assemble_public_sources.py', ['--fold', f, '--action', 'factors'], fold_dir(f) / 'assemble_factors.log')
            if stage == 'por_amplitude' and not (fold_dir(f) / 'private_POR.pt').exists():
                run_script('assemble_public_sources.py', ['--fold', f, '--action', 'private'], fold_dir(f) / 'assemble_private.log')
            write_status(status='running', current={'fold': f, 'stage': stage})
            args = ['--fold', f, '--stage', stage]
            if stage in FINAL_HEADS:
                args.append('--test')
            run_script('train_stage_public.py', args, out / 'execution.log')
            write_status(status='running', current={'fold': f, 'stage': stage, 'completed': True})
        write_status(status='running', current={'fold': f, 'completed': True})

    test_cache = torch.load(RUN / 'stages/fold_1_test/data.pt', map_location='cpu', weights_only=False)
    validation = test_cache['data']['validation']
    truth = test_cache['data']['truth'][validation]
    wells = test_cache['data']['wells'][validation]
    depth = test_cache['data']['depth'].cpu().numpy()[-len(validation):]
    columns = []
    for stage in FINAL_HEADS:
        preds = [np.load(stage_dir(f, stage) / 'test_pred.npy') for f in (1, 2, 3)]
        assert all(np.isfinite(p).all() for p in preds)
        columns.append(np.mean(preds, axis=0))
    pred = np.column_stack(columns)
    assert pred.shape == (len(truth), 3) and np.isfinite(pred).all()
    scores = {'rmse_all3': rmse3(pred, truth), **rmse_per(pred, truth),
              'tol_PHIF': tol_acc(pred[:, 0], truth[:, 0], .08),
              'tol_SW': tol_acc(np.clip(pred[:, 1], 0, 1), truth[:, 1], .05),
              'tol_VSH': tol_acc(pred[:, 2], truth[:, 2], .08)}
    per_well = {}
    for w in sorted(set(wells.tolist())):
        use = wells == w
        per_well[str(w)] = {'rmse_all3': rmse3(pred[use], truth[use]), **rmse_per(pred[use], truth[use])}
    np.savez_compressed(RUN / 'stages/public_test_predictions.npz', truth=truth, pred=pred, wells=wells, depth=depth)

    oof_pred, oof_truth, oof_fold, oof_well = [], [], [], []
    for f in (1, 2, 3):
        cache = torch.load(fold_dir(f) / 'data.pt', map_location='cpu', weights_only=False)
        oof_pred.append(np.column_stack([np.load(stage_dir(f, stage) / 'channel_pred.npy') for stage in FINAL_HEADS]))
        validation = cache['data']['validation']
        oof_truth.append(cache['data']['truth'][validation])
        oof_fold.append(np.full(len(validation), f, dtype=np.int8))
        oof_well.append(cache['data']['wells'][validation])
    oof_pred = np.concatenate(oof_pred); oof_truth = np.concatenate(oof_truth)
    oof_fold = np.concatenate(oof_fold); oof_well = np.concatenate(oof_well)
    oof_scores = {'rmse_all3': rmse3(oof_pred, oof_truth), **rmse_per(oof_pred, oof_truth)}
    np.savez_compressed(RUN / 'stages/oof_predictions.npz', truth=oof_truth, pred=oof_pred, fold=oof_fold, well=oof_well)

    root_report = json.loads((RUN / 'metrics.json').read_text(encoding='utf-8'))
    root_score = root_report['public_test']['variants']['v4_root_dual_backbone_mean']['rmse_all3']
    result = {
        'protocol': 'Full 18-stage v4 chain; final heads por_anchored, perm_split, sw_density; three-fold posterior-mean aggregation',
        'public_test': {'metrics': scores, 'per_well': per_well},
        'oof': {'metrics': oof_scores, 'rows': len(oof_truth)},
        'root_only_rmse': root_score,
        'comparison': {'rf_five_log': 0.1108188578187798, 'target_tree': 0.1078851368016146,
                       'target_tree_gate': 0.10618727913336519},
        'final_heads': list(FINAL_HEADS),
        'stage_count': len(STAGES),
        'seconds': time.perf_counter() - started,
        'no_internal_data_or_weights': True,
    }
    (RUN / 'stages/metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['RF', 'Tree', 'Tree + gate', 'v4 root', 'v4 full chain']
    values = [0.1108188578187798, 0.1078851368016146, 0.10618727913336519, root_score, scores['rmse_all3']]
    colors = ['#9ca3af', '#3973ac', '#56a374', '#b59a61', '#7856a8']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Public Volve: full 18-stage v4 evaluation')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(values): axes[0].text(i, v + .002, f'{v:.4f}', ha='center', fontsize=8)
    axes[0].set_ylim(0, max(values) * 1.18)
    ids = list(per_well)
    axes[1].plot(ids, [per_well[w]['rmse_all3'] for w in ids], marker='o', color='#7856a8', label='v4 full chain')
    root_per = root_report['public_test']['per_well']
    axes[1].plot(ids, [root_per[w]['v4_root_dual_backbone_mean']['rmse_all3'] for w in ids], marker='o', color='#b59a61', label='v4 root')
    axes[1].set_xlabel('Public test well ID'); axes[1].set_ylabel('RMSE'); axes[1].set_title('Per-well result')
    axes[1].legend(frameon=False)
    fig.savefig(RUN / 'stages/fig_full_v4_public.pdf', bbox_inches='tight')
    fig.savefig(RUN / 'stages/fig_full_v4_public.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    manifest = {
        'protocol': result['protocol'], 'seconds': result['seconds'],
        'source_scripts': {p.name: sha(p) for p in HERE.glob('*.py') if p.name in {
            'train_v4_public.py', 'train_stage_public.py', 'prepare_public_fold.py', 'run_public_stages.py',
            'train_invariant.py', 'fullcv_models_v97.py', 'fullcv_context_v97.py', 'assemble_public_sources.py'}},
        'outputs': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in [
            RUN / 'stages/metrics.json', RUN / 'stages/public_test_predictions.npz',
            RUN / 'stages/oof_predictions.npz', RUN / 'stages/fig_full_v4_public.pdf']},
        'test_limit': 'Released public test wells were previously used in other development experiments; this is a frozen public evaluation, not a new blind benchmark.',
    }
    (RUN / 'stages/experiment_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_status(status='complete', current='complete', metrics=scores)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

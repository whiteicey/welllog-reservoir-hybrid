"""Read-only progress; --final validates a fixed three-private-model assembly."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
STUDY = ROOT/'results/deep_learning/independent_sw_v9_20260928'
PREFIX = 'pure_dl_independent_sw_v9_'
MODES = ('ce', 'ce_utility')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def snapshot():
    rows = []
    for mode in MODES:
        out = ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        history = read(out/'validation_history.json') or []
        row = {'mode': mode, 'id': out.name, 'status': read(out/'runtime_status.json'),
               'config': read(out/'config.json'), 'initial': read(out/'initial_verification.json'),
               'report': read(out/'fold_5_report.json'), 'history_count': len(history),
               'last': history[-1] if history else None,
               'best_after_update': max(history, key=lambda x: x['target_score']) if history else None,
               'failure': read(out/'failure.json')}
        rows.append(row)
    out = {'observed_at': datetime.now().astimezone().isoformat(), 'runs': rows,
           'all_complete': all(x['report'] is not None for x in rows),
           'parent_manifest': 'outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json',
           'scope': 'single-target training and fixed POR/PERM substitution diagnostic until final replay',
           'no_extra_folds': True}
    if out['all_complete']:
        ce, utility = [x['report'] for x in rows]
        out['matched_difference'] = {'SW': utility['target_score']-ce['target_score'],
                                     'TOTAL_equivalent': .35*(utility['target_score']-ce['target_score'])}
    return out


def record(snap):
    from train_dual8222_heads80 import atomic_json
    atomic_json(STUDY/'progress.json', snap)
    curpath = ROOT/'docs/deep_learning/atlas/curation.json'
    cur = read(curpath)
    for row in snap['runs']:
        status = (row['status'] or {}).get('status', 'unknown')
        if row['report']:
            r = row['report']
            desc = f"SW={r['target_score']:.9f}, deltaSW={r['delta_prior_SW']:+.9f}, bestepoch={r['best_epoch']};最终头恢复通过，三模型最终磁盘重载另报。"
        elif row['last']:
            h = row['last']
            desc = f"训练快照epoch{h['epoch']}: bestSW={h['best_target_score']:.9f};尚非最终恢复成绩。"
        else:
            desc = status+'；暂无训练成绩。'
        node = cur['nodes'][row['id']]
        node['conclusion'] = desc
        node['highlight'] = status == 'running'
        path = ROOT/'outputs'/row['id']
        entry = '\n\n## 最近状态 '+snap['observed_at']+'\n'+desc+'\n'
        with (path/'PROJECTS.md').open('a', encoding='utf-8') as f:
            f.write(entry)
    atomic_json(curpath, cur)


def finalize(snap):
    if not snap['all_complete']:
        raise RuntimeError('Both predefined candidates must finish before final selection')
    from launch_independent_channels_v8 import trainers
    if trainers():
        raise RuntimeError('Training still active; do not add replay workload')
    dest = ROOT/'outputs/pure_dl_independent_sw_v9_selected_fold5'
    if dest.exists():
        raise RuntimeError('Final result already exists; do not overwrite')
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from train_dual8222_heads80 import prepare_data, SOURCE, atomic_json, atomic_array
    from train_independent_channels_v8 import REFERENCE, Evaluator as PriorEvaluator
    from independent_channels_v8 import Network as PriorModel
    from train_independent_sw_v9 import Evaluator, construct
    from gpu_job_guard_v2 import check_slot
    import train_invariant as base
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    begin = time.perf_counter()
    manifest = read(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json')
    prior_score = manifest['scores']['SW']
    winner = max(snap['runs'], key=lambda x: x['report']['target_score'])
    improved = winner['report']['target_score'] > prior_score
    print('SELECTION', json.dumps({'new_winner': winner['mode'], 'new_SW': winner['report']['target_score'],
                                  'prior_SW': prior_score, 'use_new_model': improved}), flush=True)
    master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    previous = torch.load(REFERENCE/'readout_checkpoint.pt', map_location='cpu', weights_only=False)
    # Always construct all three private models concurrently on CPU, verify
    # disjoint parameter storage, then run each on the GPU sequentially.
    models, expected, choices, ptrs = [], [], [], []
    for c, choice in enumerate(manifest['choices']):
        if c == 2 and improved:
            checkpoint = ROOT/'outputs'/winner['id']/'model_best.pt'
            cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = construct(master, previous)
            model.load_state_dict(cp['state_dict'], strict=True)
            predicted = np.load(checkpoint.parent/'channel_pred.npy')
            info = {'target': 'SW', 'mode': winner['mode'], 'checkpoint': str(checkpoint.relative_to(ROOT)),
                    'model_module': 'independent_sw_distribution_v9', 'best_epoch': cp['best_epoch']}
        else:
            checkpoint = ROOT/choice['checkpoint']
            cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = PriorModel(master, previous, choice['target'], choice['architecture'])
            model.load_state_dict(cp['state_dict'], strict=True)
            predicted = np.load(checkpoint.parent/'channel_pred.npy')
            info = dict(choice, model_module='independent_channels_v8')
        model.eval().requires_grad_(False)
        pointers = {p.data_ptr() for p in model.parameters()}
        assert all(not(pointers & prev) for prev in ptrs)
        ptrs.append(pointers)
        models.append(model)
        expected.append(predicted)
        choices.append(info)
    data = prepare_data(master)
    old = np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/fold_5_pred.npy')
    combined = np.empty_like(old)
    times = []
    for c, model in enumerate(models):
        model.cuda()
        ev = Evaluator(data, master, c, old) if c == 2 and improved else PriorEvaluator(data, master, c, old)
        torch.cuda.synchronize()
        t = time.perf_counter()
        pred, channel_score, _ = ev(model)
        torch.cuda.synchronize()
        assert np.array_equal(pred, expected[c])
        combined[:, c] = pred
        times.append({'target': choices[c]['target'], 'seconds_full_validation': time.perf_counter()-t})
        model.cpu()
    assert np.array_equal(combined[:, :2], old[:, :2])
    truth = data['truth'][data['validation']]
    scores, old_scores = base.metric(truth, combined), base.metric(truth, old)
    wells = data['wells'][data['validation']]
    perwell = []
    for w in np.unique(wells):
        mask = wells == w
        now = base.metric(truth[mask], combined[mask])
        before = base.metric(truth[mask], old[mask])
        perwell.append({'well': int(w), 'n': int(mask.sum()), 'TOTAL_delta': now['TOTAL']-before['TOTAL'],
                        'SW_delta': now['SW']-before['SW']})
    n = np.array([x['n'] for x in perwell])
    delta = np.array([x['TOTAL_delta'] for x in perwell])
    ix = np.random.default_rng(20260928).integers(0, len(n), (10000, len(n)))
    boot = (n[ix]*delta[ix]).sum(1)/n[ix].sum(1)
    reports = []
    for row in snap['runs']:
        pred = old.copy()
        pred[:, 2] = np.load(ROOT/'outputs'/row['id']/'channel_pred.npy')
        subsets = {}
        default = base.dmask(truth)[:, 2]
        for label, mask in [('true_default', default), ('true_nondefault', ~default)]:
            s1 = base.metric(truth[mask], pred[mask])['SW']
            s0 = base.metric(truth[mask], old[mask])['SW']
            subsets[label] = {'n': int(mask.sum()), 'SW_delta': s1-s0,
                              'TOTAL_contribution': float(.35*mask.mean()*(s1-s0))}
        reports.append({'mode': row['mode'], 'report': row['report'], 'population_decomposition': subsets})
    report = {'fold': 5, 'screening_fold': 5, 'scores': scores, 'reference_scores': old_scores,
              'delta_reference': scores['TOTAL']-old_scores['TOTAL'], 'selected_new_SW': improved,
              'choices': choices, 'three_private_models': True, 'no_shared_parameter_storage': True,
              'all144665rows_reloaded_exactly': True, 'POR_PERM_unchanged_exactly': True,
              'per_well': perwell, 'well_wins': int((delta > 1e-12).sum()),
              'well_losses': int((delta < -1e-12).sum()),
              'exploratory_selection_unadjusted_95': np.quantile(boot, [.025, .975]).tolist(),
              'matched_difference': snap.get('matched_difference'), 'candidate_diagnostics': reports,
              'timing': times, 'setup_and_replay_seconds': time.perf_counter()-begin,
              'completed_five_folds': False, 'remaining_folds_started': False, 'screening_threshold': 80.5,
              'screening_passed': scores['TOTAL'] > 80.5, 'development_selection': True,
              'limitations': 'Repeated development fifth; pretrained prior;SW bound and KL not separately ablated;no training-resume equivalence test.',
              'finished_at': datetime.now().astimezone().isoformat()}
    dest.mkdir()
    atomic_array(dest/'fold_5_pred.npy', combined)
    atomic_json(dest/'fold_5_report.json', report)
    atomic_json(STUDY/'assembled_report.json', report)
    recipe = {'scores': scores, 'choices': choices, 'shared_parameters_across_targets': False,
              'reference_model_construction': str((REFERENCE/'readout_checkpoint.pt').relative_to(ROOT)),
              'preprocessing_source': str(SOURCE.relative_to(ROOT)), 'old_delivery_unchanged': True,
              'complete_five_folds': False, 'no_automatic_promotion': True}
    atomic_json(dest/'CURRENT_RECIPE.json', recipe)
    atomic_json(STUDY/'CURRENT_RECIPE.json', recipe)
    atomic_json(dest/'runtime_status.json', {'status': 'complete', 'atlas_current_trial': False})
    agents = '# 目标与约束\n固定POR/PERM/SW三份私有模型，禁止逐井/逐点选模型；原数据和交付不变，无自动补折。维护PROJECTS.md记录已做、未做、计划、目标与达成后事项。\n'
    (dest/'AGENTS.md').write_text(agents, encoding='utf-8')
    (dest/'README.md').write_text('# 独立三模型固定组合\n\n'+json.dumps(recipe, ensure_ascii=False, indent=2)+'\n\n实际三模型磁盘重载验证通过；属于开发选择，不是新五折OOF。\n', encoding='utf-8')
    message = '\n\n## '+report['finished_at']+' 独立SW分布v9收尾\n'
    message += f"实际三模型TOTAL={scores['TOTAL']:.12f}，delta={report['delta_reference']:+.12f}；使用新SW={improved}。POR/PERM逐元素不变，三模型无共享存储且全部144665点重载复现。\n"
    message += '已完成v8遗留项、v8组合以及v9两项匹配训练与组合；未补折，旧交付不替换。目标完整OOF>82仍未完成。下一步按实际结果确定，不自动启动更多训练。\n'
    for path in [dest/'PROJECTS.md', STUDY/'PROJECTS.md', ROOT/'PROJECTS.md', ROOT.parent/'PROJECTS.md']:
        with path.open('a', encoding='utf-8') as f:
            f.write(message)
    final = ['# 独立SW时序分布模型v9：结果', '', '|候选|SW|替换诊断TOTAL|最佳轮次|停止轮次|', '|---|---:|---:|---:|---:|']
    for row in snap['runs']:
        r = row['report']
        final.append(f"|{row['mode']}|{r['target_score']:.9f}|{r['reference_replacement_total']:.9f}|{r['best_epoch']}|{r['stopped_epoch']}|")
    final += ['', message, '匹配损失差异：'+json.dumps(snap.get('matched_difference'), ensure_ascii=False),
              '探索性井重采样区间（未校正重复选模）：'+str(report['exploratory_selection_unadjusted_95']),
              '各候选默认/非默认人口贡献及完整证据见assembled_report.json。',
              '两个候选同时运行，单项训练时间含本项数据准备/评估/恢复；不得相加当总墙钟耗时。',
              '未独立消融对数修正上限、KL强度或初始化，不将匹配损失结论外推到所有TCN。']
    (STUDY/'FINAL.md').write_text('\n'.join(final), encoding='utf-8')
    curpath = ROOT/'docs/deep_learning/atlas/curation.json'
    cur = read(curpath)
    cur['nodes'][dest.name] = {'title': '独立三模型·SW时序分布选择',
                            'summary': 'POR/PERM保持私有模型；SW在固定两项TCN与原模型中选择。',
                            'conclusion': f"三模型实际重载TOTAL={scores['TOTAL']:.9f},delta={report['delta_reference']:+.9f};未补折。",
                            'stage': 'loss', 'highlight': True}
    for row in snap['runs']:
        cur['edges'].append({'source': row['id'], 'target': dest.name, 'type': 'evolution',
                             'note': 'SW固定候选对比后统一选择，非逐点模型选择。',
                             'evidence': 'outputs/'+dest.name+'/fold_5_report.json'})
    atomic_json(curpath, cur)
    print('ASSEMBLED', json.dumps({k: v for k, v in report.items() if k != 'per_well'}, ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    snap = snapshot()
    if args.record:
        record(snap)
    if args.final:
        finalize(snap)
    else:
        compact = {**snap, 'runs': [{k: v for k, v in row.items() if k != 'config'} for row in snap['runs']]}
        print(json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

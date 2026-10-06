"""CPU diagnosis of saved predictions, with no model/ratio/row selection.

On true continuous rows, decompose late-head gains by the old/new predicted
action class. This describes action changes, not internal gate causality.
"""
from pathlib import Path
import json
import sys

import numpy as np

S = Path(__file__).resolve().parent
R = S.parents[2]
sys.path.insert(0, str(R / 'experiments/pure_dl'))
from fullcv_context_v97 import original_population
from train_invariant import dmask, metric, TS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


population = original_population()
cases = []
for path in sorted(S.glob('fold_*/*_PREFIX_SNAPSHOT.json')):
    complete = read(path)
    if not complete['complete8stage_single_fold']:
        continue
    fold, identity = complete['fold'], complete['identity']
    truth = population['truth'][population['fold'] == fold - 1]
    own = R / 'outputs/pure_dl_two_seed_fullcv_v370' / f'fold_{fold}' / identity
    for stage, parent, target in [('POR341', 'POR151', 'POR'), ('PERM336', 'PERM154', 'PERM')]:
        terminal = read(S / f'fold_{fold}' / (identity + '_' + stage + '_TERMINAL.json'))
        gain = terminal['stage_gains'][-1]
        column = TS.index(target)
        before, after = truth.copy(), truth.copy()
        before[:, column] = np.load(own / parent / 'channel_pred.npy')
        after[:, column] = np.load(own / stage / 'channel_pred.npy')
        assert np.isfinite(before).all() and np.isfinite(after).all()
        assert abs(metric(truth, after)[target] - metric(truth, before)[target] - gain['actual_target_gain']) < 1e-9
        continuous = ~dmask(truth)[:, column]
        old_default = dmask(before)[:, column]
        new_default = dmask(after)[:, column]
        total = int(continuous.sum())
        assert total == gain['strata']['CONT']['rows'] and total > 0
        groups = []
        coverage = np.zeros(len(truth), np.int8)
        for old in (False, True):
            for new in (False, True):
                mask = continuous & (old_default == old) & (new_default == new)
                rows = int(mask.sum())
                coverage[mask] += 1
                label = ('DEFAULT' if old else 'CONT') + '_to_' + ('DEFAULT' if new else 'CONT')
                delta = metric(truth[mask], after[mask])[target] - metric(truth[mask], before[mask])[target] if rows else 0.
                groups.append(dict(predicted_action_transition=label, rows=rows,
                                   group_target_delta=delta, weighted_CONT_delta=delta * rows / total))
        assert (coverage[continuous] == 1).all() and (coverage[~continuous] == 0).all()
        summed = sum(group['weighted_CONT_delta'] for group in groups)
        assert abs(summed - gain['strata']['CONT']['delta']) < 1e-9
        cases.append(dict(fold=fold, identity=identity, stage=stage, target=target,
                          actual_target_gain=gain['actual_target_gain'],
                          CONT_delta=gain['strata']['CONT']['delta'], CONT_rows=total,
                          predicted_DEFAULT_before=int((continuous & old_default).sum()),
                          predicted_DEFAULT_after=int((continuous & new_default).sum()),
                          groups=groups))

result = dict(cases=cases, CPU_only=True, extra_training_stages=0,
              scope='Completed8-stage cases only; historical fifth excluded; identical saved predictions and original score',
              action_class_definition='Original dmask applied to predictions: POR<=0.1, PERM<=0.01, SW>=99.9',
              caution='Transition groups use validation truth for diagnosis only; no rowwise model selection, weight changes, or inference rule',
              causal_gate_explanation_proven=False)
(S / 'CONTINUOUS_ACTION_DIAGNOSIS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
lines = ['# Continuous-row action diagnosis', '',
         'Saved predictions only; no training, weight change or rowwise choice. Original continuous labels define analysis rows.',
         'Predicted DEFAULT/CONT uses the original action threshold, not an inspected internal gate.',
         'Group contributions sum exactly to each recorded CONT delta. Internal gate causality remains unproven.', '',
         '| Fold | Model | Stage | CONT delta | Predicted DEFAULT before/after | CONT-to-CONT contribution | CONT-to-DEFAULT contribution |',
         '|---:|---|---|---:|---:|---:|---:|']
for case in cases:
    groups = {group['predicted_action_transition']: group for group in case['groups']}
    lines.append(f"| {case['fold']} | {case['identity']} | {case['stage']} | {case['CONT_delta']:+.9f} | "
                 f"{case['predicted_DEFAULT_before']}/{case['predicted_DEFAULT_after']} | "
                 f"{groups['CONT_to_CONT']['weighted_CONT_delta']:+.9f} | {groups['CONT_to_DEFAULT']['weighted_CONT_delta']:+.9f} |")
lines += ['', 'Full four-group breakdown is in CONTINUOUS_ACTION_DIAGNOSIS.json.',
          'Cases may share folds/training wells; this is descriptive evidence, not independent replication or a causal ablation.']
(S / 'CONTINUOUS_ACTION_DIAGNOSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(dict(completed_model_cases=len(cases) // 2, diagnosed_stages=len(cases), CPU_only=True,
                      third_fold=[case for case in cases if case['fold'] == 3])))

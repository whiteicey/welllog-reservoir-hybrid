"""CPU-only actual composition and strata for completed v297 arms."""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pure_dl'))
import train_invariant as base

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

data = torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt', map_location='cpu', weights_only=False)
truth = np.asarray(data['truth'][data['validation']])
default = base.dmask(truth)[:, 2]
old = json.loads((ROOT/'results/deep_learning/perm_sparse_reacquisition_v291_20261002/CURRENT_EIGHT_STAGE_RECIPE.json').read_text(encoding='utf-8-sig'))
diagnostic = {}
paths = {
    'SW252': ROOT/'outputs/pure_dl_sw_mass_cdf_v252_fold5/conditional/channel_pred.npy',
    'SW289': Path(old['heads']['SW']['checkpoint']).parent/'channel_pred.npy',
}
# Resolve the retained conditional control from its study instead of guessing aliases.
if not paths['SW252'].exists():
    candidates = list((ROOT/'outputs').glob('*v252_fold5/*/channel_pred.npy'))
    if len(candidates) != 1:
        raise RuntimeError(f'Ambiguous SW252 control: {candidates}')
    paths['SW252'] = candidates[0]
for arm in ('well', 'reference'):
    out = ROOT/f'outputs/pure_dl_sw_raw_well_condition_v297_fold5/{arm}'
    if not (out/'report.json').exists():
        continue
    report = json.loads((out/'report.json').read_text(encoding='utf-8-sig'))
    assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    channel = np.load(out/'channel_pred.npy')
    pred = np.column_stack([np.load(Path(old['heads'][t]['checkpoint']).parent/'channel_pred.npy') for t in ('POR','PERM')] + [channel])
    assert pred.shape == (144665, 3) and np.isfinite(pred).all()
    scores = base.metric(truth, pred)
    assert abs(scores['SW']-report['target_score']) < 1e-9
    recipe = copy.deepcopy(old)
    recipe.update(scores=scores, strict_808_qualified=scores['TOTAL']>80.8,
                  gain_vs_previous_optional8=scores['TOTAL']-old['scores']['TOTAL'],
                  gain_vs_default_five=scores['TOTAL']-80.61247541799393,
                  paired_reference_final_pending=not (ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/reference/report.json').exists(),
                  verification='actual144665 composition; own-best RAM replay/frozen checks, not independent disk reload')
    recipe.pop('matched_pair_delta', None)
    recipe.pop('paired_window_final_pending', None)
    recipe['stages'][-1] = f'SW297{arm}'
    recipe['heads']['SW'] = dict(checkpoint=str(out/'model_best.pt'), sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest())
    np.save(STUDY/f'{arm}_optional_eight_stage_pred.npy', pred)
    recipe['pred_sha256'] = hashlib.sha256((STUDY/f'{arm}_optional_eight_stage_pred.npy').read_bytes()).hexdigest()
    write(STUDY/f'{arm.upper()}_EIGHT_STAGE_RECIPE.json', recipe)
    paths[f'v297_{arm}'] = out/'channel_pred.npy'
for name, path in paths.items():
    pred = truth.copy()
    pred[:,2] = np.load(path)
    diagnostic[name] = dict(SW=base.metric(truth,pred)['SW'], default_SW=base.metric(truth[default],pred[default])['SW'], continuous_SW=base.metric(truth[~default],pred[~default])['SW'], default_rows=int(default.sum()), continuous_rows=int((~default).sum()), source=str(path))
write(STUDY/'STRATA_DIAGNOSTIC.json', diagnostic)
print(json.dumps(diagnostic))

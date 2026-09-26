"""Training-only diagnostic: fixed rows, no predictions or model selection exported."""
import importlib
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as inv
from train import TS, metric, dmask
from src.utils import DEFAULTS, BOUNDS, kernel_matrix

ROOT = Path(__file__).resolve().parents[2]
MODELS = {'pure_dl_batchensemble_invariant': 'train_batchensemble_invariant',
          'pure_dl_tabm32': 'train_tabm32', 'pure_dl_tabm32_wide': 'train_tabm32_wide'}
parser = argparse.ArgumentParser()
parser.add_argument('--runs', nargs='+', choices=list(MODELS), default=list(MODELS)[:2])
parser.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_summary/training_fit_diagnostic.json')
args = parser.parse_args()
torch.set_num_threads(3)
rec = inv.load_split(ROOT.parent/'data', 'train')
y_all = [frame[TS].to_numpy(np.float64) for _, frame in rec]
valid = [np.isfinite(y).all(1) for y in y_all]
y = np.concatenate([v[m] for v, m in zip(y_all, valid)])
well = np.concatenate([np.full(m.sum(), i) for i, m in enumerate(valid)])
row_ids = [np.flatnonzero(well == i) for i in range(80)]
source = ROOT/'outputs/pure_dl_batchensemble_invariant'
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    reference = torch.load(source/'fold_1.pt', map_location='cpu', weights_only=True)
heldout = set(reference['config']['outer_wells']['0'])
train_wells = [i for i, (name, _) in enumerate(rec) if name not in heldout]
assert len(train_wells) == 64
rows = np.random.default_rng(0).choice(np.flatnonzero(np.isin(well, train_wells)), 16384, replace=False)
assert all(rec[i][0] not in heldout for i in np.unique(well[rows]))
x, _, fm, fs, edges = inv.prepare_features(rec, train_wells, valid, row_ids)
results = []
for name in args.runs:
    module = MODELS[name]
    mod = importlib.import_module(module)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(ROOT/'outputs'/name/'fold_1.pt', map_location='cpu', weights_only=True)
    assert ck['config']['outer_wells'] == reference['config']['outer_wells']
    assert np.array_equal(fm, ck['feature_median']) and np.array_equal(fs, ck['feature_scale'])
    assert np.array_equal(edges, ck['ple_edges_normalized'])
    model = mod.Network(edges).cuda().eval(); model.load_state_dict(ck['state_dict'])
    actions, kernels = [], []
    for j, target in enumerate(TS):
        sup = np.asarray(ck['centers'][j])
        action = np.linspace(sup[0], sup[-1], 800) if j == 2 else np.geomspace(max(sup[0]*.9, .001), sup[-1]*1.1, 800)
        action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
        actions.append(action)
        kernels.append(torch.as_tensor(kernel_matrix(action, np.r_[DEFAULTS[target], sup], target).T, device='cuda', dtype=torch.float32))
    prediction = np.empty((len(rows), 3)); gates = np.empty_like(prediction)
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            rr = rows[start:start+128]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                go, raw = model(x[torch.as_tensor(rr, device='cuda')][None].expand(mod.MEMBERS, -1, -1))
            logw, mu, sd = mod.mixture_params(raw)
            for j in range(3):
                cp = mod.conditionals(logw[:, :, j].exp(), mu[:, :, j], sd[:, :, j],
                    torch.as_tensor(ck['density_edges'][j], device='cuda', dtype=torch.float32))
                gate, _, full = mod.joint_average(go[:, :, j].float().sigmoid(), cp)
                prediction[start:start+len(rr), j] = actions[j][(full@kernels[j]).argmax(-1).cpu().numpy()]
                gates[start:start+len(rr), j] = gate.cpu().numpy()
    target_gate = dmask(y[rows]); clipped = np.clip(gates, 1e-8, 1-1e-8)
    results.append({'model': name, 'training_subset_score': metric(y[rows], prediction),
        'training_ensemble_gate_BCE': float(-(target_gate*np.log(clipped)+(1-target_gate)*np.log(1-clipped)).mean()),
        'recorded_outer_fold1_score': json.loads((ROOT/'outputs'/name/'fold_1_report.json').read_text())['scores']})
    print(json.dumps(results[-1]), flush=True)
    del model; torch.cuda.empty_cache()
out = args.out
out.write_text(json.dumps({'scope': 'Optimistic in-sample diagnostic on16384 fixed rows from64 fold1 training wells; not CV evidence. No thresholds, weights or output predictions selected.',
                          'sampling_seed': 0, 'excluded_outer_wells': sorted(heldout), 'results': results}, indent=2), encoding='utf-8')

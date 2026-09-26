"""Training-fold-only default-state counts; no fitting or validation scoring."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
source = root / 'outputs/pure_dl_global_cosine_sequence/oof.npz'
out = root / 'outputs/training_gate_state_diagnostic'
out.mkdir(exist_ok=True)
with np.load(source) as pack:
    selected = pack['fold'] != 4
    y = pack['truth'][selected]
    wells = pack['well'][selected]
assert len(y) == 578891 and len(np.unique(wells)) == 64
default = np.column_stack((y[:, 0] <= .1, y[:, 1] <= .01, y[:, 2] >= 99.9))
state = default.astype(np.int64) @ np.array([1, 2, 4])
counts = np.bincount(state, minlength=8)
observed = counts / len(y)
marginal = default.mean(0)
bits = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
independent = np.prod(np.where(bits, marginal, 1 - marginal), axis=1)
positive = observed > 0
kl = float(np.sum(observed[positive] * np.log(observed[positive] / independent[positive])))
result = {
    'scope': 'Only fixed fifth-fold training population; no validation labels, predictions or fitting',
    'training_points': len(y), 'training_wells': len(np.unique(wells)),
    'channels': ['POR', 'PERM', 'SW'], 'bit_one': 'default state, not anomaly',
    'default_marginal': marginal.tolist(),
    'states': [{'bits': row.astype(int).tolist(), 'count': int(counts[i]),
                'fraction': float(observed[i]), 'independent_fraction': float(independent[i])}
               for i, row in enumerate(bits)],
    'unconditional_total_correlation_nats': kl,
    'well_counts': [{'well': str(w), 'counts': np.bincount(state[wells == w], minlength=8).tolist()}
                    for w in np.unique(wells)],
    'limitation': 'Unconditional dependence does not prove residual dependence given inputs or benefit from joint gating. Official utility remains target-separable.'
}
(out / 'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'well_counts'}, indent=2))

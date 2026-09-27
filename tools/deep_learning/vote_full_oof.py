"""Full-OOF official score with equal seed-vote over decoded probability grids.

Same-fold checkpoints across seeds share identical centers (seed only changes RNG),
so bin-space probability averaging is valid. Fold-specific centers differ, but each
row belongs to exactly one fold so we decode per-fold with that fold's grid.
"""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'pure_dl'))
import train_invariant as base  # noqa: E402

TARGETS = ['POR', 'PERM', 'SW']

def main():
    seed_dirs = sys.argv[1:]
    assert len(seed_dirs) >= 2
    oof0 = np.load(Path(seed_dirs[0]) / 'oof.npz')
    truth, fold = oof0['truth'], oof0['fold']
    n = len(truth)
    for sd in seed_dirs[1:]:
        oof = np.load(Path(sd) / 'oof.npz')
        assert np.array_equal(oof['truth'], truth) and np.array_equal(oof['fold'], fold)

    n_seeds = len(seed_dirs)
    pred = np.full_like(truth, np.nan)

    for f in [1, 2, 3, 4, 5]:
        idx = np.flatnonzero(fold == (f - 1))
        # use fold's shared centers from first seed
        z0 = np.load(Path(seed_dirs[0]) / f'fold_{f}_distributions.npz', allow_pickle=True)
        for j, t in enumerate(TARGETS):
            ct = z0[f'{t}_centers']
            action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0] * .9, .001), ct[-1] * 1.1, 800)
            action = np.unique(np.r_[base.DEFAULTS[t], np.clip(action, *base.BOUNDS[t])])
            U = base.kernel_matrix(action, np.r_[base.DEFAULTS[t], ct], t).T.astype(np.float64)
            probs = []
            gates = []
            for sd in seed_dirs:
                z = np.load(Path(sd) / f'fold_{f}_distributions.npz', allow_pickle=True)
                probs.append(z[f'{t}_prob'].astype(np.float64))
                gates.append(z['gate'][:, j:j+1])
                z.close()
            P = np.mean(probs, 0)
            G = np.mean(gates, 0)
            joint = np.concatenate([G, (1 - G) * P], axis=1)
            pred[idx, j] = action[(joint @ U).argmax(1)]
        z0.close()

    print('SEED_ENSEMBLE_FULL_OOF', json.dumps(base.metric(truth, pred)))
    for sd in seed_dirs:
        rpt = json.loads((Path(sd) / 'report.json').read_text())
        print(f"  single {sd}: TOTAL={rpt['oof_scores']['TOTAL']:.6f}")


if __name__ == '__main__':
    main()

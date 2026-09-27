"""Average fold-5 decoded probability grids across seeds and rescore with the official metric."""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(REPO / 'experiments' / 'pure_dl'))
import train_invariant as base  # noqa: E402

TARGETS = ['POR', 'PERM', 'SW']


def load_pack(out_dir):
    z = np.load(Path(out_dir) / 'fold_5_distributions.npz', allow_pickle=True)
    pack = {k: z[k] for k in z.files}
    z.close()
    return pack


def decode_and_score(packs, truth, validation):
    # Average decoded probability grids across seeds (equal weights).
    n = len(validation)
    avg_probs = {}
    for name in TARGETS:
        stack = np.stack([p[f'{name}_prob'].astype(np.float64) for p in packs], 0)
        avg_probs[name] = stack.mean(0)
    gates = np.stack([p['gate'].astype(np.float64) for p in packs], 0).mean(0)
    centers = [packs[0][f'{name}_centers'] for name in TARGETS]

    pred = np.full_like(truth, np.nan)
    for j, name in enumerate(TARGETS):
        ct = centers[j]
        action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0] * .9, .001), ct[-1] * 1.1, 800)
        action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
        utility = base.kernel_matrix(action, np.r_[base.DEFAULTS[name], ct], name).T
        gate = gates[:, j:j + 1]
        joint = np.concatenate([gate, (1 - gate) * avg_probs[name]], -1)
        pred[validation, j] = action[(joint @ utility).argmax(-1)]
    return base.metric(truth[validation], pred[validation])


def main():
    outs = sys.argv[1:]
    assert len(outs) >= 2, 'need at least 2 output dirs'
    packs = [load_pack(o) for o in outs]
    # Truth and validation indices from any pack's oof.npz
    oof = np.load(Path(outs[0]) / 'oof.npz')
    truth, fold = oof['truth'], oof['fold']
    validation = np.flatnonzero(fold == 4)
    for p in packs:
        assert np.array_equal(p['indices'], validation), 'validation indices mismatch'
    scores = decode_and_score(packs, truth, validation)
    print('SEED_ENSEMBLE', json.dumps(scores))
    for o in outs:
        r = json.loads((Path(o) / 'fold_5_report.json').read_text())
        print(f"  single {o}: TOTAL={r['scores']['TOTAL']:.6f} best_epoch={r['best_epoch']}")


if __name__ == '__main__':
    main()

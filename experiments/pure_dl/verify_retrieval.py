"""Verify fold-local label-memory membership and saved neural distributions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from retrieval_network import self_check, validate_memory_split

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_retrieval')
args = parser.parse_args()
config = json.loads((args.out/'config.json').read_text())
oof = np.load(args.out/'oof.npz')
self_check()
for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == expected, name
assert config['test_access'] is False and config['training_query_own_well_masked'] is True
findings = []
for f in range(5):
    ck = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=True)
    train = np.flatnonzero(oof['fold'] != f)
    val = np.flatnonzero(oof['fold'] == f)
    memory = np.asarray(ck['memory_rows'])
    validate_memory_split(memory, train, val)
    assert np.array_equal(ck['outer_training_rows'], train)
    assert len(memory) == len(set(memory)) == 4096
    ids, counts = np.unique(oof['well'][memory], return_counts=True)
    assert len(ids) == 64 and np.all(counts == 64)
    assert set(ids) == set(oof['well'][train])
    assert not set(ids) & set(oof['well'][val])
    assert np.array_equal(ck['memory_wells'], oof['well'][memory])
    base = Path(ck['base_checkpoint'])
    assert base.resolve() == (ROOT/'outputs/pure_dl_ple'/f'fold_{f+1}.pt').resolve()
    assert hashlib.sha256(base.read_bytes()).hexdigest() == config['base_checkpoint_sha256'][str(f+1)]
    with np.load(args.out/f'fold_{f+1}_distributions.npz') as dist:
        assert np.array_equal(dist['indices'], val)
        gate = dist['gate']
        assert gate.shape == (len(val), 3) and np.isfinite(gate).all()
        assert ((gate >= 0) & (gate <= 1)).all()
        for target in ['POR', 'PERM', 'SW']:
            p = dist[target+'_prob'].astype(np.float32)
            c = dist[target+'_centers']
            assert p.shape == (len(val), 256) and c.shape == (256,)
            assert np.isfinite(p).all() and np.isfinite(c).all() and (p >= 0).all()
            assert np.all(np.diff(c) > 0)
            assert np.max(np.abs(p.sum(-1)-1)) < .001
    findings.append({'fold': f+1, 'memory_rows': len(memory), 'memory_wells': len(ids),
                     'outer_validation_wells_excluded': True, 'base_checkpoint_sha256_matches': True})
result = {'passed': True, 'folds': findings,
          'checks': ['saved memory is an exact subset of the expected outer training wells',
                     '64 memory points from each of 64 training wells; no validation memory',
                     'saved source code and same-fold base checkpoint digests match',
                     'source module self-check: same-well label perturbation invariance and held-out memory rejection',
                     'saved probability dimensions, normalization, finite values and row alignment']}
(args.out/'memory_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

"""Audit source provenance and fold-local statistics for the NLL continuation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_batchensemble_nll60')
args = parser.parse_args()
cfg = json.loads((args.out/'config.json').read_text())
source = Path(cfg['warmstart_source'])
assert source.resolve() == (ROOT/'outputs/pure_dl_batchensemble_invariant').resolve()
assert json.loads((source/'verification.json').read_text())['passed'] is True
assert cfg['source_training_epochs'] == 40 and cfg['epochs'] == 20 and cfg['total_training_epochs'] == 60
assert cfg['continuation_learning_rate'] == .0003
source_candidates = [Path(__file__).parent/'finetune_batchensemble_nll.py',
                     args.out/'training_source/finetune_batchensemble_nll.py']
verified_source = next((p for p in source_candidates if p.exists() and
                        hashlib.sha256(p.read_bytes()).hexdigest() == cfg['continuation_source_sha256']), None)
assert verified_source is not None, 'No exact training source matching the recorded SHA256'
for relative, expected in cfg['dependency_sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
findings = []
for f in range(5):
    path = source/f'fold_{f+1}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg['warmstart_sha256'][str(f+1)]
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        a = torch.load(path, map_location='cpu', weights_only=True)
        b = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=True)
    for key in ['feature_median', 'feature_scale', 'ple_edges_normalized', 'regression_median',
                'regression_scale', 'centers', 'density_edges']:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), (f, key)
    assert a['config']['outer_wells'] == b['config']['outer_wells']
    assert any(not torch.equal(value, b['state_dict'][key])
               for key, value in a['state_dict'].items() if value.is_floating_point())
    findings.append({'fold': f+1, 'source_digest_matches': True,
                     'input_and_target_statistics_equal_audited_same_fold_source': True,
                     'outer_well_split_unchanged': True, 'weights_updated': True})
result = {'passed': True, 'folds': findings, 'source_code_digests_match': True,
          'verified_training_source': str(verified_source),
          'source_training_epochs': 40, 'continuation_epochs': 20,
          'scope': 'Checks source identity, same-fold fitted statistics and well split; official scores checked separately by verify.py'}
(args.out/'warmstart_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

"""CPU-only completed-stage evidence; never rank an incomplete seed model."""
from pathlib import Path
import json, sys, time
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pure_dl'))
from train import metric, dmask

seed = int(sys.argv[1])
own = ROOT / 'outputs/pure_dl_complete_seeds_v362_fold5' / f'seed_{seed}'
data = torch.load(own / 'SOURCE/prepared_data.pt', map_location='cpu', weights_only=False)
truth = np.asarray(data['truth'][data['validation']]); del data
source = np.load(own / 'SOURCE/fold_5_pred.npy')
assert source.shape == truth.shape == (144665, 3)
defaults = dmask(truth)
targets = ('POR', 'PERM', 'SW')
weights = (.3, .35, .35)
parents = dict(POR113='SOURCE', POR151='POR113', PERM154='SOURCE',
               SW104='SOURCE', POR341='POR151', PERM336='PERM154', SW351='SW104')
predictions = {}; stages = []
assembled = source.copy()
for name, parent in parents.items():
    path = own / ('SW351_fixed' if name == 'SW351' else name)
    if not (path / 'report.json').is_file() or not (path / 'TIMING.json').is_file():
        break
    timing = json.loads((path / 'TIMING.json').read_text(encoding='utf-8-sig'))
    if not timing.get('terminal') or timing.get('exit_code') != 0:
        break
    assert not psutil.pid_exists(timing['pid']), (name, timing['pid'])
    report = json.loads((path / 'report.json').read_text(encoding='utf-8-sig'))
    target = report['target']; c = targets.index(target)
    after = np.load(path / 'channel_pred.npy')
    assert after.shape == (144665,) and np.isfinite(after).all()
    before = source[:, c] if parent == 'SOURCE' else predictions[parent]
    test = truth.copy(); test[:, c] = before
    before_score = metric(truth, test)[target]
    test[:, c] = after; after_score = metric(truth, test)[target]
    assert abs(after_score - report['target_score']) < 1e-9
    strata = {}
    for kind, mask in [('DEFAULT', defaults[:, c]), ('CONT', ~defaults[:, c])]:
        test = truth[mask].copy(); test[:, c] = before[mask]
        a = metric(truth[mask], test)[target]
        test[:, c] = after[mask]; b = metric(truth[mask], test)[target]
        strata[kind] = dict(before=a, after=b, delta=b-a)
    stages.append(dict(stage=name, target=target, predecessor=parent,
                       actual_target_gain=after_score-before_score,
                       actual_TOTAL_contribution=weights[c]*(after_score-before_score),
                       before=before_score, after=after_score,
                       optimizer_gain=report['against_initial'],
                       best_epoch=report['best_epoch'], stop_epoch=report['stopped_epoch'],
                       predictions_identical_to_predecessor=np.array_equal(before, after),
                       strata=strata))
    predictions[name] = after; assembled[:, c] = after
complete_model = (own/'COMPLETE_MODEL.json').is_file() and len(stages) == 7
scope = ('Complete eight-stage CPU prediction readback, not independent model disk reload'
         if complete_model else 'Completed-stage prefix only, not complete eight-stage model or seed ranking')
result = dict(seed=seed, observed_epoch=time.time(), completed_stages=1+len(stages),
              planned_stages=8, stages=stages, prefix_scores=metric(truth, assembled),
              score_scope=scope,
              complete_model=complete_model, screening_fold=5,
              complete_five_folds=False, CPU_only=True, extra_training_stages=0)
(HERE / f'SEED{seed}_STAGE_SNAPSHOT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

"""Independently reconstruct complete saved point table from channel evidence history."""
import argparse
import json
import numpy as np
import torch
from verify_family_mixture_conflict_trace import verify


def audit(out):
    trace = verify(out, 5)
    ck = torch.load(out/'fold_5.pt', map_location='cpu', weights_only=False)
    table = np.load(out/'fold_5_global_point_weights.npz')
    history = np.load(out/'fold_5_conflict_weights.npz')
    ids = table['raw_ids']; observed = history['raw_ids']
    assert np.array_equal(ids, np.sort(ck['training_raw_rows']))
    assert not np.intersect1d(ids, ck['validation_raw_rows']).size
    assert np.array_equal(np.sort(observed),ids), 'Full training coverage'
    ix = np.searchsorted(ids, observed)
    assert np.array_equal(ids[ix], observed)
    weight = np.ones(len(ids), np.float32)
    reason = np.zeros(len(ids), np.uint8)
    count = np.zeros(len(ids), np.int32)
    epoch = np.full(len(ids), -1, np.int32)
    evidence = np.ones((len(ids), 3), np.float32)
    for e, q in zip(history['epochs'], history['weights']):
        new = np.min(q, axis=1)
        reason[ix] = np.where(new > weight[ix], 3, np.where(new < 1, 2, 1))
        weight[ix] = new; evidence[ix] = q; epoch[ix] = e; count[ix] += 1
    for name, value in [('weight',weight),('reason',reason),('updates',count),
                        ('last_epoch',epoch),('channel_evidence_weight',evidence)]:
        assert np.array_equal(table[name], value), name
    probe = np.load(out/'fold_5_dynamics/training_dynamics_history.npz')
    assert np.array_equal(table['well_ids'][ix],probe['well_ids'])
    assert np.array_equal(table['block_ids'][ix],probe['block_ids'])
    rho=probe['por_family_t3']
    assert rho.shape==(len(probe['epochs']),len(observed))
    assert np.isfinite(rho).all() and ((rho>=-1e-6)&(rho<=1+1e-6)).all()
    summaries=json.loads((out/'fold_5_family_history.json').read_text())
    assert len(summaries)==len(probe['epochs'])
    for j,summary in enumerate(summaries):
        assert summary['epoch']==int(probe['epochs'][j]) and summary['points']==len(observed)
        for group,subset in [('all',rho[j]),('nondefault_POR',rho[j][~probe['default'][:,0]])]:
            stats=summary[group]
            assert stats['count']==len(subset)
            assert abs(stats['mean']-float(subset.mean()))<1e-8
            assert abs(stats['std']-float(subset.std()))<1e-8
            assert np.allclose(stats['quantiles_01_10_50_90_99'],np.quantile(subset,[.01,.1,.5,.9,.99]),atol=1e-8,rtol=0)
            assert abs(stats['below05']-float((subset<.05).mean()))<1e-12
            assert abs(stats['above95']-float((subset>.95).mean()))<1e-12

    result = {'passed':True, 'training_points':len(ids), 'evaluated_points':len(observed),
              'unobserved_points':int((count==0).sum()), 'final_downweighted_points':int((weight<1).sum()),
              'minimum_point_weight':float(weight.min()), 'channel_policy_replay_passed':trace['passed'],'training_only_family_summary_replayed':True,
              'scope':'Exact final table replay; final observation may occur after last optimizer step. Table is diagnostic training state, not inference input.'}
    (out/'global_point_table_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    audit(p.parse_args().out)

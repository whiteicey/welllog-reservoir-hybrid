"""One predeclared, label-independent pure-DL fifth-fold consensus blend."""
import json
from pathlib import Path

import numpy as np
import train_invariant as base


ROOT = base.ROOT
POST = ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'
CAL = ROOT/'outputs/pure_dl_paired_calibrated_view_sequence'
OUT = ROOT/'outputs/pure_dl_fixed_consensus_blend_fold5'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(POST/'oof.npz') as a, np.load(CAL/'oof.npz') as b:
        assert np.array_equal(a['truth'], b['truth'])
        assert np.array_equal(a['fold'], b['fold'])
        assert np.array_equal(a['well'], b['well'])
        fold = a['fold'].copy()
        truth = a['truth'].copy()
        post = a['pred'][fold == 4].copy()
        cal = b['pred'][fold == 4].copy()
    assert len(post) == len(cal) == 144665
    assert np.isfinite(post).all() and np.isfinite(cal).all()
    assert np.array_equal(post, np.load(POST/'fold_5_pred.npy'))
    assert np.array_equal(cal, np.load(CAL/'fold_5_pred.npy'))
    blended = post.copy()
    p_default = base.dmask(post)
    c_default = base.dmask(cal)
    consensus_counts = {}
    for j, name in enumerate(base.TS):
        both_nondefault = ~p_default[:, j] & ~c_default[:, j]
        consensus_counts[name] = {
            'both_nondefault': int(both_nondefault.sum()),
            'both_default': int((p_default[:, j] & c_default[:, j]).sum()),
            'default_disagreement': int((p_default[:, j] ^ c_default[:, j]).sum()),
        }
        transform = base.TRANSFORM[j]
        inverse = base.INVERSE[j]
        middle = .5 * (transform(post[both_nondefault, j]) +
                       transform(cal[both_nondefault, j]))
        blended[both_nondefault, j] = np.clip(inverse(middle), *base.BOUNDS[name])
    assert np.isfinite(blended).all()
    assert np.array_equal(base.dmask(blended), p_default)
    y = truth[fold == 4]
    post_score = base.metric(y, post)
    cal_score = base.metric(y, cal)
    score = base.metric(y, blended)
    result = {
        'rule': 'fixed equal midpoint in each training target coordinate when both actions are nondefault; otherwise retain posterior action',
        'selection_label_scope': 'none; fold5 truth used only after the fixed blend to score',
        'screening_fold': 5,
        'rows': len(y),
        'threshold_strict_gt': 80.4,
        'passed': bool(score['TOTAL'] > 80.4),
        'posterior_scores': post_score,
        'calibrated_scores': cal_score,
        'blended_scores': score,
        'delta_vs_posterior_total': score['TOTAL'] - post_score['TOTAL'],
        'delta_vs_calibrated_total': score['TOTAL'] - cal_score['TOTAL'],
        'consensus_counts': consensus_counts,
    }
    np.save(OUT/'fold_5_pred.npy', blended)
    (OUT/'screening_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

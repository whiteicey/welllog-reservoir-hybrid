"""Input-only within-well conditional ranks and deviations given GR rank.

These are statistical input contexts, not inferred geological labels.
"""
import numpy as np

CURVES = ('PE', 'SP', 'CAL', 'AC', 'DEN', 'CNL', 'RXO', 'RT')
FEATURE_NAMES = tuple(c + '_' + s for c in CURVES for s in
                      ('GR_CONDITIONAL_RANK', 'GR_CONDITIONAL_DEVIATION'))


def conditional_features(frame):
    n = len(frame)
    result = np.zeros((n, 16), dtype=np.float32)
    gr = frame['GR'].to_numpy(np.float64)
    good_gr = np.isfinite(gr)
    if good_gr.sum() < 16:
        return result
    ordered_gr = np.sort(gr[good_gr])
    u = np.zeros(n)
    u[good_gr] = (np.searchsorted(ordered_gr, gr[good_gr], 'left') +
                  np.searchsorted(ordered_gr, gr[good_gr], 'right')) / (2 * len(ordered_gr))
    anchors = np.linspace(0, 1, 9)
    for j, curve in enumerate(CURVES):
        raw = frame[curve].to_numpy(np.float64).copy()
        if curve in ('RXO', 'RT'):
            raw = np.log1p(np.maximum(raw, 0))
        rows = np.flatnonzero(good_gr & np.isfinite(raw))
        if len(rows) < 16:
            continue
        rows = rows[np.argsort(raw[rows], kind='stable')]
        value, rank = raw[rows], u[rows]
        weights = np.exp(-0.5 * ((rank[None, :] - anchors[:, None]) / 0.125) ** 2)
        prefix = np.concatenate([np.zeros((9, 1)), np.cumsum(weights, axis=1)], axis=1)
        total = prefix[:, -1:]
        left = np.searchsorted(value, value, 'left')
        right = np.searchsorted(value, value, 'right')
        # Mid-CDF handles equal values identically, independent of row order.
        cdf = (prefix[:, left] + prefix[:, right]) / (2 * total)
        quantiles = np.stack([
            value[np.minimum(np.searchsorted(prefix[k, 1:] / total[k, 0], [.25, .5, .75]), len(value)-1)]
            for k in range(9)
        ])
        pos = rank * 8
        lo = np.minimum(pos.astype(np.int64), 7)
        mix = pos - lo
        col = np.arange(len(rows))
        conditional_rank = (1-mix)*cdf[lo, col] + mix*cdf[lo+1, col]
        q = (1-mix[:, None])*quantiles[lo] + mix[:, None]*quantiles[lo+1]
        global_iqr = np.percentile(value, 75) - np.percentile(value, 25)
        scale = np.maximum(q[:, 2]-q[:, 0], max(global_iqr*.05, 1e-6))
        result[rows, 2*j] = 2*conditional_rank - 1
        result[rows, 2*j+1] = np.tanh((value-q[:, 1])/scale)
    return result

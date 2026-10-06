"""Fold-local persistent table keyed by original raw row, never by window position."""
from pathlib import Path
import numpy as np


class GlobalPointWeights:
    def __init__(self, nraw, training_raw, starts):
        self.allowed = np.zeros(nraw, bool)
        self.allowed[np.asarray(training_raw, dtype=np.int64)] = True
        self.weight = np.ones(nraw, np.float32)
        self.channel_evidence_weight = np.ones((nraw, 3), np.float32)
        self.last_epoch = np.full(nraw, -1, np.int32)
        self.updates = np.zeros(nraw, np.int32)
        self.reason = np.zeros(nraw, np.uint8)  # 0 unobserved, 1 neutral, 2 conflict, 3 recovering
        self.starts = np.asarray(starts)

    def update(self, raw_ids, channel_weights, epoch):
        ids = np.asarray(raw_ids, dtype=np.int64)
        q = np.asarray(channel_weights, dtype=np.float32)
        assert ids.ndim == 1 and len(np.unique(ids)) == len(ids)
        assert ((ids >= 0) & (ids < len(self.weight))).all()
        assert self.allowed[ids].all(), 'Only this fold training points can update'
        assert q.shape == (len(ids), 3) and np.isfinite(q).all()
        assert ((q >= .1) & (q <= 1)).all()
        assert (epoch > self.last_epoch[ids]).all()
        old = self.weight[ids].copy()
        current = q.min(axis=1)  # One point penalty; no repeated well/block/channel multiplication.
        self.channel_evidence_weight[ids] = q
        self.weight[ids] = current
        self.reason[ids] = np.where(current > old, 3, np.where(current < 1, 2, 1))
        self.last_epoch[ids] = epoch
        self.updates[ids] += 1
        return current.copy()

    def save(self, path):
        ids = np.flatnonzero(self.allowed)
        well = np.searchsorted(self.starts[1:], ids, side='right')
        path = Path(path)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, raw_ids=ids, well_ids=well,
                block_ids=(ids-self.starts[well])//128, weight=self.weight[ids],
                channel_evidence_weight=self.channel_evidence_weight[ids],
                last_epoch=self.last_epoch[ids], updates=self.updates[ids], reason=self.reason[ids])
        temp.replace(path)

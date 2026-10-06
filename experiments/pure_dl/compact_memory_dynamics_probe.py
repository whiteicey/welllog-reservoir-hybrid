"""Exact float32 CDF change in RAM; persist compact final diagnostics only."""
import json
import numpy as np

from training_por_t3_full import TrainingDynamicsProbe


class CompactMemoryDynamicsProbe(TrainingDynamicsProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The original probe uses a 1.78 GiB file-backed float32 CDF. Its
        # first record never reads the prior CDF; every row is written once.
        # Keep the identical float32 update in RAM without a D-drive temp file.
        self.previous_cdf = np.empty((len(self.raw), 3, 256), dtype=np.float32)

    def finalize(self):
        if not self.epochs:
            return {'records': 0}
        self.previous_cdf = None
        # The live policy has already consumed every full-point probe. Keep
        # the final point state, row identity and timing, avoiding a ~0.7 GiB
        # full-trajectory archive. This does not change training or scoring.
        np.savez_compressed(
            self.out/'training_dynamics_final.npz',
            epochs=np.asarray(self.epochs), raw_ids=self.raw, well_ids=self.well,
            block_ids=self.block, default=self.default,
            **{key: values[-1] for key, values in self.history.items()},
        )
        result = {
            'records': len(self.epochs), 'points': len(self.raw),
            'total_probe_seconds': self.probe_seconds,
            'scope': 'All labeled training rows; exact float32 prior CDF in RAM. Final point state saved; full historical trajectory intentionally omitted.',
        }
        (self.out/'training_dynamics_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

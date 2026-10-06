"""Atomic outer-fold source recovery without a multi-GB CDF checkpoint.

The previous observation model regenerates the CDF after an explicit resume.
Only training inputs/labels enter that regeneration. Observation statistics and
weights are separate atomic files; a last state references committed epochs.
"""
from pathlib import Path
import random
import json
import numpy as np
import torch
from training_por_t3_full import TrainingDynamicsProbe


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def atomic_npz(path, **values):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


class CompactProbe(TrainingDynamicsProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_model = None

    def record(self, model, x, epoch):
        result = super().record(model, x, epoch)
        self.history = {k: [v[-1]] for k, v in self.history.items()}
        self.observed_model = cpu_state(model)
        return result

    def commit(self, weights):
        atomic_npz(self.out/f'observation_{self.epochs[-1]:03d}.npz',
                   epoch=self.epochs[-1], weights=weights.astype(np.float32),
                   **{k: v[-1] for k, v in self.history.items()})

    def recovery_state(self):
        return dict(epochs=self.epochs.copy(), observed_model=self.observed_model,
                    seconds=self.probe_seconds, peak=self.probe_peak_gpu_mib)

    def recover(self, saved, model, x):
        if not saved['epochs']:
            return
        # Discard the possibly interrupted mutable CDF and reconstruct it from
        # the committed observation predictor. record preserves training RNG.
        current = cpu_state(model)
        model.load_state_dict(saved['observed_model'])
        super().record(model, x, saved['epochs'][-1])
        model.load_state_dict(current)
        self.observed_model = saved['observed_model']
        self.epochs = saved['epochs'].copy()
        with np.load(self.out/f'observation_{self.epochs[-1]:03d}.npz') as z:
            self.history = {k: [z[k].copy()] for k in (
                'nll','conditional_entropy','target_tail','gate_probability','cdf_l1_change')}
        self.probe_seconds = saved['seconds']
        self.probe_peak_gpu_mib = saved['peak']

    def export_quality(self, path):
        weights = []
        for epoch in self.epochs:
            with np.load(self.out/f'observation_{epoch:03d}.npz') as z:
                assert int(z['epoch']) == epoch
                weights.append(z['weights'].copy())
        atomic_npz(path, epochs=np.asarray(self.epochs), raw_ids=self.raw,
                   weights=np.stack(weights))

    def finalize(self):
        if self.previous_cdf is not None:
            self.previous_cdf.flush()
            self.previous_cdf._mmap.close()
            self.previous_cdf = None
            (self.out/'previous_cdf.tmp').unlink()
        result = dict(records=len(self.epochs), points=len(self.raw),
                      total_probe_seconds=self.probe_seconds,
                      scope='Full own-fold training population; atomic observation files; CDF temporary removed.')
        (self.out/'training_dynamics_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

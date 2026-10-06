"""Held matched POR341 replacement: training-only continuous scalar auxiliary."""
import torch
from torch import nn
from torch.nn import functional as F
from por_coupling_conditional_rps_v341 import (
    ROOT, SOURCE, PARENT, PARENT_SHA, load_data, Evaluator,
    Network as Parent, channel_objective as parent_objective,
)

STUDY = ROOT / 'results/deep_learning/por_continuous_aux_v367_20261002'
OUT = ROOT / 'outputs/pure_dl_por_continuous_aux_v367_fold5'
STAGES = {mode: ('POR', 256, False, 0., 3e-4) for mode in ('aux0', 'aux01')}


class Network(Parent):
    def __init__(self, master, metadata, mode):
        super().__init__(master, metadata, 'conditional')
        self.continuous_coefficient = {'aux0': 0., 'aux01': .1}[mode]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261367)
            self.continuous_readout = nn.Linear(48, 1)
        self.continuous_features = None
        self.last_continuous_aux = 0.
        self.last_main_loss = 0.

    def groups(self):
        groups = super().groups()
        readout = next(g for g in groups if g['name'] == 'POR_direct_categorical_readout')
        readout['params'] = list(readout['params']) + list(self.continuous_readout.parameters())
        return groups

    def initialize_priors(self, data):
        # The parent counts its own134752 trainable parameters. Do not alter its
        # frozen source or checks when adding49 matched training-only parameters.
        self.continuous_readout.requires_grad_(False)
        try:
            info = super().initialize_priors(data)
        finally:
            self.continuous_readout.requires_grad_(True)
        info.update(
            continuous_auxiliary_parameters=49,
            new_active_parameters=134801,
            active_parameters=134801,
            continuous_coefficient=self.continuous_coefficient,
            continuous_target='original train-fitted transformed POR z, real nondefault only',
            auxiliary_normalization='Q-weighted CONT mean times mean(Q)/3',
            auxiliary_is_predictive_distribution_mean=False,
            auxiliary_used_in_inference=False,
            auxiliary_beta=1.,
            parent341_learned_checkpoint_loaded=False,
            same_structure_and_initialization_pair=True,
            primary_control='V367 aux0 same134801 capacity/init and80/pat10->14; aux01 changes coefficient to0.1 only',
            secondary_control='Historical341 score reference only; historical stopping budget differs',
        )
        return info

    def forward(self, *args, **kwargs):
        self.continuous_features = None
        output = super().forward(*args, **kwargs)
        if self.training:
            self.continuous_features = self.aux_features[-1]
        return output


def make_model(stage, master, metadata):
    return Network(master, metadata, stage)


def channel_objective(output, labels, default, mask, quality, target_index, mode, model, well_ids=None):
    assert target_index == 0 and mode in STAGES
    features = model.continuous_features
    assert features is not None and model.training
    model.continuous_features = None
    main, likelihood, rps = parent_objective(
        output, labels, default, mask, quality, 0, 'conditional', model, well_ids=well_ids,
    )
    prediction = model.continuous_readout(features[mask].float()).squeeze(-1)
    target = labels[mask][:, 0].float()
    q = quality[mask].detach().float()
    continuous_weight = q * (~default[mask][:, 0]).float()
    row_loss = F.smooth_l1_loss(prediction, target, reduction='none', beta=1.)
    auxiliary = (continuous_weight * row_loss).sum() / continuous_weight.sum().clamp_min(1e-12)
    model.last_continuous_aux = float(auxiliary.detach())
    model.last_main_loss = float(main.detach())
    return main + model.continuous_coefficient * auxiliary * q.mean() / 3, likelihood, rps

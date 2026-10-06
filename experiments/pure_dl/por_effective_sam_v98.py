"""Matched AdamW/SAM coadaptation of the retained POR routes and location branch.

Both arms start at the selected v31 function in the fixed v96 composition.
No new capacity, backbone updates, output voting, or other-target losses.
"""
import copy
import hashlib
import json

import numpy as np
import torch
from torch import nn

from por_reparam_v31 import Network as Parent, ROOT, Evaluator, load_data, TARGETS, channel_objective

STUDY = ROOT/'results/deep_learning/por_effective_sam_v98_20260930'
PREVIOUS = ROOT/'results/deep_learning/sw_factor_sam_v96_20260930'
MODES = ('adamw', 'sam')
LEARNING_RATE = 3e-5
SAM_RHO = .05


def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')


class Network(Parent):
    def __init__(self, target, master, modes=MODES):
        super().__init__(target, master, modes=('anchored',))
        recipe = json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        choice = next(c for c in recipe['choices'] if c['target'] == 'POR')
        path = ROOT/choice['checkpoint']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['target'] == 'POR' and saved['mode'] == 'anchored'
        self.heads['anchored'].load_state_dict(saved['delta_state'], strict=True)
        template = copy.deepcopy(self.heads['anchored'])
        self.requires_grad_(False)
        heads = {}
        for mode in modes:
            assert mode in MODES
            head = copy.deepcopy(template)
            head.requires_grad_(False)
            head.mode = 'coadapt'
            for unit in (head.route28.unit, *head.route29.branches):
                unit.input.requires_grad_(True)
                unit.output.requires_grad_(True)
            for readout in head.readouts:
                readout.residual.delta_in.requires_grad_(True)
                readout.residual.delta_out.requires_grad_(True)
            heads[mode] = head
        self.heads = nn.ModuleDict(heads)
        self.factor = 'location'
        self.study_metadata = dict(parent_checkpoint=choice['checkpoint'],
                                   parent_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   warm_start='complete selected v31 anchored function',
                                   updated='existing bounded routes and existing amplitude location correction',
                                   frozen='private dual backbones, calibration, original point32 branch, scale, weights and all anchors',
                                   new_parameters=0, sampler_seed=20261010,
                                   optimizer_only_ablation=True)
        self.train(False)

    def trainable_groups(self, mode):
        groups = {'routes': [], 'location': []}
        for name, parameter in self.heads[mode].named_parameters():
            if parameter.requires_grad:
                group = 'routes' if name.startswith(('route28.', 'route29.')) else 'location'
                groups[group].append(parameter)
        assert all(groups.values())
        return [dict(params=params, lr=LEARNING_RATE, name=name) for name, params in groups.items()]


def factor_scope(model, data, mode):
    return dict(updated='existing routes and existing location correction only',
                route_bounds=[model.heads[mode].route28.unit.bound,
                              *[branch.bound for branch in model.heads[mode].route29.branches]],
                backbone_and_other_factor_integrity='whole frozen-state equality checked after actual training',
                optimizer='AdamW' if mode == 'adamw' else 'two-pass SAM around AdamW')

"""Training-well curve prototypes condition only the second expert's input calibration."""
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_calibrated_view_sequence import CalibratedMember
from paired_well_view_sequence import WellViewMember, RAW_CURVES
from paired_detached_router_sequence import routed_merge


PROTOTYPES = 12


class FaciesCalibratedMember(CalibratedMember):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, child in original.named_children():
            self.add_module(name, child)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in original.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        with torch.random.fork_rng(devices=[]):
            self.facies_calibration = nn.Linear(PROTOTYPES, 62)
            nn.init.zeros_(self.facies_calibration.weight)
            nn.init.zeros_(self.facies_calibration.bias)

    def forward(self, features, side, relative_depth, valid, auxiliary=False, absolute_depth=None):
        clean = features.masked_fill(~valid[..., None], 0)
        context = clean.index_select(-1, self.conditioning_indices)
        delta = self.calibration(context) + self.facies_calibration(side.float().masked_fill(~valid[..., None], 0))
        gamma, beta = delta.chunk(2, -1)
        numeric = clean.index_select(-1, self.numeric_indices)
        replacement = (numeric * (1 + gamma.tanh()) + beta.tanh()).masked_fill(~valid[..., None], 0).to(clean.dtype)
        updated = clean.index_copy(-1, self.numeric_indices, replacement)
        return WellViewMember.forward(self, updated, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[1] = FaciesCalibratedMember(self.members[1])

    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        assert features.shape[-1] == 206 + PROTOTYPES
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        side = features[..., 206:]
        return [self.members[0](original, relative_depth, valid, auxiliary, absolute_depth),
                self.members[1](original, side, relative_depth, valid, auxiliary, absolute_depth)], original

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members, original = self._members(features, relative_depth, valid, False, absolute_depth)
        logits = self.route_logits(original, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members, original = self._members(features, relative_depth, valid, auxiliary, absolute_depth)
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(original, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)

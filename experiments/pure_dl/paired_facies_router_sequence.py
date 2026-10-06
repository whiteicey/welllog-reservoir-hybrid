"""Train-well unsupervised curve-pattern prototypes augment the posterior router."""
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES
from paired_detached_router_sequence import routed_merge


PROTOTYPES = 12


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.facies_router = nn.Linear(PROTOTYPES, 6)
            nn.init.zeros_(self.facies_router.weight)
            nn.init.zeros_(self.facies_router.bias)

    def route_logits(self, features, valid, main_outputs):
        assert features.shape[-1] == 206 + PROTOTYPES
        base = super().route_logits(features[..., :206], valid, main_outputs)
        side = features[..., 206:].float().masked_fill(~valid[..., None], 0)
        correction = self.facies_router(side).reshape(*features.shape[:2], 3, 2)
        return base + correction.masked_fill(~valid[..., None, None], 0)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        assert features.shape[-1] == 206 + PROTOTYPES
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, absolute_depth=absolute_depth) for m in self.members]
        logits = self.route_logits(features, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        assert features.shape[-1] == 206 + PROTOTYPES
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, auxiliary, absolute_depth) for m in self.members]
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(features, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)

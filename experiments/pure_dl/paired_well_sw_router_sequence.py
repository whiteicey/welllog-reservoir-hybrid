"""Input-only whole-well summaries condition only SW expert routing."""
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_sw_prior_sequence import RAW_CURVES, well_context_features
from paired_detached_router_sequence import routed_merge


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.sw_well_router = nn.Sequential(nn.Linear(36, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.sw_well_router[-1].weight)
            nn.init.zeros_(self.sw_well_router[-1].bias)

    def route_logits(self, features, valid, main_outputs):
        assert features.shape[-1] == 242
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        logits = super().route_logits(original, valid, main_outputs)
        context = features[..., 206:].float().masked_fill(~valid[..., None], 0)
        delta = self.sw_well_router(context).masked_fill(~valid[..., None], 0)
        sw = logits[..., 2, :] + delta
        return torch.cat([logits[..., :2, :], sw.unsqueeze(-2)], -2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, absolute_depth=absolute_depth) for m in self.members]
        logits = self.route_logits(features, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, auxiliary, absolute_depth) for m in self.members]
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(features, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)

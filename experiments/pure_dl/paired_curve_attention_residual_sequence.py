"""Nine-curve pointwise attention residual on the full-feature expert."""
import torch
from torch import nn
from torch.nn import functional as F

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES
from paired_detached_router_sequence import routed_merge


class CurveAttentionResidual(nn.Module):
    def __init__(self, raw_indices, relative_indices, missing_indices):
        super().__init__()
        for name, indices in [('raw', raw_indices), ('relative', relative_indices), ('missing', missing_indices)]:
            assert len(indices) == 9 and len(set(indices)) == 9
            self.register_buffer(name+'_indices', torch.tensor(indices, dtype=torch.long))
        self.token_input = nn.Linear(3, 16)
        self.identity = nn.Parameter(torch.randn(9, 16)*.02)
        self.norm = nn.LayerNorm(16)
        self.attention = nn.MultiheadAttention(16, 2, dropout=0., batch_first=True)
        self.mix = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.output = nn.Linear(16, 206)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, valid):
        assert features.shape[-1] == 206
        stacked = torch.stack([features.index_select(-1, self.raw_indices),
                               features.index_select(-1, self.relative_indices),
                               features.index_select(-1, self.missing_indices)], -1).float()
        batch, length = valid.shape
        tokens = self.token_input(stacked.reshape(batch*length, 9, 3)) + self.identity
        tokens = self.norm(tokens)
        attended = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        pooled = (tokens + attended).mean(1)
        correction = self.output(self.mix(F.gelu(pooled))).reshape(batch, length, 206)
        return correction.masked_fill(~valid[..., None], 0)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices,
                 conditioning_indices, raw_indices, relative_indices, missing_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.curve_adapter = CurveAttentionResidual(raw_indices, relative_indices, missing_indices)

    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        original = features.to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        correction = self.curve_adapter(original, valid)
        full = (original.float() + correction.float()).to(original.dtype)
        return [self.members[0](full, relative_depth, valid, auxiliary, absolute_depth),
                self.members[1](original, relative_depth, valid, auxiliary, absolute_depth)], original

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

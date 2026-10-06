"""Pointwise nine-curve attention corrects only each expert's SW main head."""
import torch
from torch import nn
from torch.nn import functional as F

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES
from paired_detached_router_sequence import routed_merge


class SWCurveAttention(nn.Module):
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
        self.output = nn.Linear(16, 9)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, valid):
        source = torch.stack([features.index_select(-1, self.raw_indices),
                              features.index_select(-1, self.relative_indices),
                              features.index_select(-1, self.missing_indices)], -1).float()
        batch, length = valid.shape
        tokens = self.norm(self.token_input(source.reshape(batch*length, 9, 3)) + self.identity)
        attended = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        delta = self.output(self.mix(F.gelu((tokens+attended).mean(1))))
        return delta.reshape(batch, length, 9).masked_fill(~valid[..., None], 0)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices,
                 conditioning_indices, raw_indices, relative_indices, missing_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.sw_adapters = nn.ModuleList([
                SWCurveAttention(raw_indices, relative_indices, missing_indices) for _ in range(2)])

    def _correct(self, output, index, features, valid):
        gate, raw = output
        delta = self.sw_adapters[index](features, valid)
        sw_gate = (gate[..., 2] + delta[..., 0]).to(gate.dtype)
        corrected_gate = torch.cat([gate[..., :2], sw_gate[..., None]], -1)
        sw = raw[..., 2, :, :]
        sw_mean = (sw[..., 1, :] + delta[..., 1:]).to(raw.dtype)
        corrected_sw = torch.stack([sw[..., 0, :], sw_mean, sw[..., 2, :]], -2)
        corrected_raw = torch.cat([raw[..., :2, :, :], corrected_sw.unsqueeze(-3)], -3)
        return corrected_gate, corrected_raw

    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        result = []
        for index, member in enumerate(self.members):
            output = member(features, relative_depth, valid, auxiliary, absolute_depth)
            if auxiliary:
                result.append((self._correct(output[0], index, features, valid), output[1], output[2]))
            else:
                result.append(self._correct(output, index, features, valid))
        return result

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members = self._members(features, relative_depth, valid, False, absolute_depth)
        logits = self.route_logits(features, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members = self._members(features, relative_depth, valid, auxiliary, absolute_depth)
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(features, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)

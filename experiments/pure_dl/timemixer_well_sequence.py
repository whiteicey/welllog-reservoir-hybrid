"""TimeMixer-inspired well-depth encoder, not an official forecasting reproduction.

Masked trend/residual decomposition and directional scale mixing are adapted
to synchronous POR/PERM/SW prediction. There is no future-target input.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import DensityHead, objective


def masked_average(x, valid, width, stride=1, padding=0):
    mask = valid[:, None].to(x.dtype)
    numerator = F.avg_pool1d(x.transpose(1, 2)*mask, width, stride, padding)
    denominator = F.avg_pool1d(mask, width, stride, padding)
    value = numerator/denominator.clamp_min(1e-6)
    return value.transpose(1, 2), denominator.squeeze(1) > 0


class DepthProjection(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(source, target), nn.GELU(), nn.Linear(target, target))

    def forward(self, value):
        return self.layers(value.transpose(1, 2)).transpose(1, 2)


class DecomposedScaleBlock(nn.Module):
    def __init__(self, lengths=(256, 128, 64), width=64):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in lengths])
        self.detail_down = nn.ModuleList([DepthProjection(lengths[i], lengths[i+1]) for i in range(2)])
        self.trend_up = nn.ModuleList([DepthProjection(lengths[i+1], lengths[i]) for i in range(2)])
        self.channel = nn.ModuleList([nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width)) for _ in lengths])
        self.down_gain = nn.Parameter(torch.full((2,), .1))
        self.up_gain = nn.Parameter(torch.full((2,), .1))
        self.output_gain = nn.Parameter(torch.full((3, width), .1))

    def forward(self, levels, masks):
        detail, trend = [], []
        for value, mask, norm in zip(levels, masks, self.norms):
            normalized = norm(value)*mask[:, :, None]
            smooth, _ = masked_average(normalized, mask, 17, padding=8)
            smooth = smooth*mask[:, :, None]
            trend.append(smooth); detail.append(normalized-smooth)
        mixed_detail = [detail[0]]
        for i in range(2):
            mixed_detail.append((detail[i+1]+self.down_gain[i]*self.detail_down[i](mixed_detail[-1]))*masks[i+1][:, :, None])
        mixed_trend = [None, None, trend[2]]
        for i in (1, 0):
            mixed_trend[i] = (trend[i]+self.up_gain[i]*self.trend_up[i](mixed_trend[i+1]))*masks[i][:, :, None]
        return [(levels[i]+self.output_gain[i]*self.channel[i](mixed_detail[i]+mixed_trend[i]))*masks[i][:, :, None] for i in range(3)]


class Network(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(features+2, 64), nn.GELU())
        self.mixers = nn.ModuleList([DecomposedScaleBlock(), DecomposedScaleBlock()])
        self.lift = nn.ModuleList([nn.Identity(), DepthProjection(128, 256), DepthProjection(64, 256)])
        self.point = nn.Sequential(nn.Linear(features, 128), nn.GELU(), nn.Linear(128, 64))
        self.combine = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1))
        self.main_head = DensityHead(128, initial_means)
        self.aux_mid = DensityHead(64, initial_means)
        self.aux_coarse = DensityHead(64, initial_means)

    def forward(self, features, relative_depth, valid, auxiliary=False):
        assert features.shape[1] == 256
        mask = valid[:, :, None].to(features.dtype)
        point = features*mask
        value = self.embedding(torch.cat([point, relative_depth[:, :, None]*mask, mask], -1))*mask
        levels, masks = [value], [valid]
        for _ in range(2):
            value, flag = masked_average(levels[-1], masks[-1], 2, stride=2)
            levels.append(value); masks.append(flag)
        for block in self.mixers:
            levels = block(levels, masks)
        combined = torch.cat([self.lift[i](levels[i]) for i in range(3)]+[self.point(point)], -1)
        output = self.main_head(self.combine(combined).transpose(1, 2))
        if not auxiliary:
            return output
        return output, self.aux_mid(levels[1].transpose(1, 2)), self.aux_coarse(levels[2].transpose(1, 2))


def self_test():
    torch.set_num_threads(2); torch.manual_seed(107)
    model = Network(features=12).eval()
    x = torch.randn(2, 256, 12); depth = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[0, :120] = False; valid[0, 136:] = False
    with torch.no_grad():
        output = model(x, depth, valid)
        changed = x.clone(); changed[~valid] += 1000
        assert all(torch.equal(a, b) for a, b in zip(output, model(changed, depth, valid)))
        other = x.clone(); other[1] += 100
        assert all(torch.equal(a[0], b[0]) for a, b in zip(output, model(other, depth, valid)))
        constant = torch.ones(2, 256, 3)*7
        smooth, mask = masked_average(constant, valid, 17, padding=8)
        assert torch.allclose(smooth[mask], torch.full_like(smooth[mask], 7), atol=1e-5)
    target = torch.randn(2, 256, 3); default = torch.rand(2, 256, 3) > .4
    outputs = model(x, depth, valid, auxiliary=True)
    loss = objective(outputs, target, default, valid, deep_supervision=True); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.mixers[0].detail_down[0].layers[0].weight.grad.abs().sum() > 0
    assert model.mixers[0].trend_up[0].layers[0].weight.grad.abs().sum() > 0
    return {'passed': True, 'padded_inputs_masked': True, 'different_wells_isolated': True,
            'masked_average_preserves_constant_signal': True, 'both_mixing_directions_receive_gradients': True,
            'parameters': sum(p.numel() for p in Network().parameters()), 'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))

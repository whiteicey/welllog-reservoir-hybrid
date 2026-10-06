"""Dense well-depth U-Net with a pointwise MLP and optional deep supervision.

No learned numerical encoding, label retrieval, or ensemble-member branches.
Every input window belongs to exactly one well.
"""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


class GatedBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3*dilation,
                                   dilation=dilation, groups=channels)
        self.expand = nn.Conv1d(channels, channels*2, 1)
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(.1)
        self.gain = nn.Parameter(torch.full((1, channels, 1), .1))

    def forward(self, x):
        value, gate = self.expand(self.depthwise(self.norm(x))).chunk(2, dim=1)
        return x+self.gain*self.project(self.dropout(F.gelu(value)*gate.sigmoid()))


class DensityHead(nn.Module):
    def __init__(self, width, initial_means=None):
        super().__init__()
        self.gate = nn.Conv1d(width, 3, 1)
        self.density = nn.Conv1d(width, 72, 1)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            bias = self.density.bias.view(3, 3, 8)
            if initial_means is not None:
                bias[:, 1].copy_(torch.as_tensor(initial_means))
            bias[:, 2].fill_(math.log(math.expm1(.5-.025)))

    def forward(self, x):
        return self.gate(x).transpose(1, 2), self.density(x).transpose(1, 2).reshape(len(x), -1, 3, 3, 8)


class Network(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.stem = nn.Conv1d(features+2, 64, 1)
        self.enc0 = GatedBlock(64)
        self.down1 = nn.Conv1d(64, 96, 4, stride=2, padding=1)
        self.enc1 = GatedBlock(96)
        self.down2 = nn.Conv1d(96, 128, 4, stride=2, padding=1)
        self.enc2 = GatedBlock(128, dilation=2)
        self.down3 = nn.Conv1d(128, 192, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(GatedBlock(192, 1), GatedBlock(192, 2), GatedBlock(192, 4))
        self.merge2 = nn.Conv1d(192+128, 128, 1)
        self.dec2 = GatedBlock(128)
        self.merge1 = nn.Conv1d(128+96, 96, 1)
        self.dec1 = GatedBlock(96)
        self.merge0 = nn.Conv1d(96+64, 64, 1)
        self.dec0 = GatedBlock(64)
        self.point = nn.Sequential(nn.Conv1d(features, 128, 1), nn.GELU(), nn.Conv1d(128, 64, 1))
        self.combine = nn.Sequential(nn.Conv1d(128, 128, 1), nn.GELU(), nn.Dropout(.1))
        self.main_head = DensityHead(128, initial_means)
        self.aux_mid = DensityHead(96, initial_means)
        self.aux_coarse = DensityHead(128, initial_means)

    def forward(self, features, relative_depth, valid, auxiliary=False):
        # features(B,L,206), relative_depth/valid(B,L). Padded inputs are masked.
        mask = valid[:, None].to(features.dtype)
        point = features.transpose(1, 2)*mask
        x = torch.cat([point, relative_depth[:, None]*mask, mask], dim=1)
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = self.dec2(self.merge2(torch.cat([F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)


def head_loss(output, target_z, target_default, supervised):
    gate, raw = output
    length = target_z.shape[1]
    if gate.shape[1] != length:
        repeat = length//gate.shape[1]
        assert repeat*gate.shape[1] == length
        gate = gate.repeat_interleave(repeat, dim=1)
        raw = raw.repeat_interleave(repeat, dim=1)
    assert supervised.shape == target_z.shape[:2] and bool(supervised.any())
    # Select first: invalid/out-of-well labels have exactly no loss or gradients.
    gate = gate[supervised].float(); raw = raw[supervised].float()
    truth = target_z[supervised]; default = target_default[supervised].float()
    weight = raw[:, :, 0].log_softmax(-1)
    mu = raw[:, :, 1]; sd = F.softplus(raw[:, :, 2])+.025
    nll = -(weight-.5*((truth[:, :, None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    terms = []
    for j in range(3):
        active = default[:, j] < .5
        terms.append(nll[active, j].mean() if bool(active.any()) else nll[:, j].sum()*0)
    return F.binary_cross_entropy_with_logits(gate, default)+sum(terms)/3


def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    main, mid, coarse = outputs
    return (head_loss(main, target_z, target_default, supervised)
            +.2*head_loss(mid, target_z, target_default, supervised)
            +.1*head_loss(coarse, target_z, target_default, supervised))


def window_indices(centers, lower, upper, length=256, anchors=None):
    assert length % 8 == 0
    anchors = centers if anchors is None else anchors
    index = centers[:, None]+torch.arange(-length//2, length//2, device=centers.device)
    valid = (index >= lower[anchors, None]) & (index < upper[anchors, None])
    safe = torch.maximum(lower[anchors, None], torch.minimum(index, upper[anchors, None]-1))
    return safe, valid


def self_test():
    torch.set_num_threads(2); torch.manual_seed(54)
    model = Network(features=12).eval()
    x = torch.randn(2, 256, 12); d = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[:, :32] = False
    targets = torch.randn(2, 256, 3); defaults = torch.rand(2, 256, 3) > .4
    supervised = valid.clone(); supervised[:, :64] = False; supervised[:, 192:] = False
    outputs = model(x, d, valid, auxiliary=True)
    assert outputs[0][0].shape == (2, 256, 3)
    assert outputs[0][1].shape == (2, 256, 3, 3, 8)
    assert outputs[1][0].shape[1] == 128 and outputs[2][0].shape[1] == 64
    loss = objective(outputs, targets, defaults, supervised, True)
    altered = targets.clone(); altered[~supervised] = 1e6
    altered_d = defaults.clone(); altered_d[~supervised] = ~altered_d[~supervised]
    assert torch.equal(loss, objective(outputs, altered, altered_d, supervised, True))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.aux_coarse.density.weight.grad.abs().sum() > 0
    changed = x.clone(); changed[~valid] += 1e5
    with torch.no_grad():
        a = model(x, d, valid); b = model(changed, d, valid)
        assert all(torch.equal(u, v) for u, v in zip(a, b))
        other = x.clone(); other[1] += 100
        c = model(other, d, valid)
        assert all(torch.equal(u[0], v[0]) for u, v in zip(a, c))
    low = torch.tensor([0]*10+[10]*10); high = low+10
    idx, mask = window_indices(torch.tensor([0, 9, 10, 19]), low, high)
    assert ((idx < 10) == (torch.tensor([0, 9, 10, 19])[:, None] < 10)).all()
    assert (mask.sum(-1) == 10).all()
    idx, mask = window_indices(torch.tensor([64, 74]), low, high, anchors=torch.tensor([9, 19]))
    assert (idx[0] < 10).all() and (idx[1] >= 10).all()
    assert (mask[:, 64:192].sum(-1) == 10).all()
    return {'passed': True, 'dense_shapes': True, 'padded_input_invariant': True,
            'masked_labels_cannot_change_loss': True, 'whole_well_window_isolation': True,
            'other_batch_well_invariant': True, 'all_loss_gradients_finite': True,
            'auxiliary_head_receives_gradients': True}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))

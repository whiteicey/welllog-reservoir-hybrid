"""Aligned-depth feature exchange between two original global-cosine members."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_backbone_sequence import Network as Baseline, objective, merge_predictions


def forward_features(member, features, relative_depth, valid, absolute_depth):
    # Explicit equivalent of the inherited absolute_depth_sequence forward up
    # to the head inputs. RelativeOnlyStem itself removes the absolute channel.
    mask = valid[:, None].to(features.dtype)
    point = features.transpose(1, 2) * mask
    assert absolute_depth is not None and absolute_depth.shape == valid.shape
    x = torch.cat([point, relative_depth[:, None]*mask, mask,
                   absolute_depth[:, None]*mask], dim=1)
    e0 = member.enc0(member.stem(x))
    e1 = member.enc1(member.down1(e0))
    e2 = member.enc2(member.down2(e1))
    h = member.bottleneck(member.down3(e2))
    h = member.dec2(member.merge2(torch.cat([
        F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
    coarse = h
    h = member.dec1(member.merge1(torch.cat([
        F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
    mid = h
    h = member.dec0(member.merge0(torch.cat([
        F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
    return member.combine(torch.cat([h, member.point(point)], 1)), mid, coarse


class CrossAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.input = nn.Linear(128, 32)
        self.output = nn.Linear(32, 128)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, valid):
        value = features.transpose(1, 2)
        correction = self.output(F.gelu(self.input(self.norm(value))))
        return correction.masked_fill(~valid[..., None], 0).transpose(1, 2)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        with torch.random.fork_rng(devices=[]):
            self.cross_readout = nn.ModuleList([CrossAdapter(), CrossAdapter()])

    def member_predictions(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        representations = [forward_features(member, features, relative_depth, valid, absolute_depth)
                           for member in self.members]
        # Simultaneous exchange: each reads the uncorrected other feature.
        fine = [representations[i][0] + self.cross_readout[i](representations[1-i][0], valid)
                for i in range(2)]
        outputs = []
        for i, member in enumerate(self.members):
            main = member.main_head(fine[i])
            if auxiliary:
                outputs.append((main, member.aux_mid(representations[i][1]),
                                member.aux_coarse(representations[i][2])))
            else:
                outputs.append(main)
        return outputs

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        outputs = self.member_predictions(features, relative_depth, valid, auxiliary, absolute_depth)
        return outputs if auxiliary else merge_predictions(outputs)

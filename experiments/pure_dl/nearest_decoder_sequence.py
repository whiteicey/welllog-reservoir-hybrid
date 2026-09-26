"""Nearest-neighbor decoding with unchanged patch hierarchy backbone."""
import torch
from torch.nn import functional as F
from patch_hierarchy_sequence import Network as Baseline

class Network(Baseline):
    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        # features(B,L,206), relative_depth/valid(B,L). Padded inputs are masked.
        mask = valid[:, None].to(features.dtype)
        point = features.transpose(1, 2)*mask
        assert absolute_depth is not None and absolute_depth.shape == valid.shape
        x = torch.cat([point, relative_depth[:, None]*mask, mask, absolute_depth[:, None]*mask], dim=1)
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = self.dec2(self.merge2(torch.cat([F.interpolate(h, size=e2.shape[-1], mode='nearest'), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([F.interpolate(h, size=e1.shape[-1], mode='nearest'), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([F.interpolate(h, size=e0.shape[-1], mode='nearest'), e0], 1)))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

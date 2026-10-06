"""Residual exchange of global patch scores between attention heads."""
import torch
from torch import nn
from cosine_patch_sequence import cosine_scores
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


class HeadExchange(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.exchange = nn.Parameter(torch.zeros(original.heads, original.heads))

    def forward(self, x, valid):
        old = self.original
        batch, length, width = x.shape
        assert length <= old.maximum_tokens
        z = old.norm(x).masked_fill(~valid[..., None], 0)
        q, k, v = old.qkv(z).reshape(batch, length, 3, old.heads, width // old.heads).permute(2, 0, 3, 1, 4)
        scores = cosine_scores(q, k, old.log_scale)
        pos = torch.arange(length, device=x.device)
        scores = scores + old.relative_bias[:, pos[:, None] - pos[None, :] + old.maximum_tokens - 1]
        # Mix finite, unmasked scores first; padding is excluded afterwards.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = scores.float() + torch.einsum('hg,bgij->bhij', self.exchange.float(), scores.float())
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        attended = (scores.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(batch, length, width)
        return old.project(attended).masked_fill(~valid[..., None], 0)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        hierarchy = self.members[0].stem.hierarchy
        hierarchy.global_patches = HeadExchange(hierarchy.global_patches)

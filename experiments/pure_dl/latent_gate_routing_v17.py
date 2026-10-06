"""Parameter-matched SW latent-memory routing experiment.

Both arms retain exactly the v16 latent parameters. The local path shares the
pre/post local convolution weights with the global path. Only the feature read
by the default-state head changes; the 256-bin density always reads global
memory. No model, parameter, or hidden feature is shared with POR/PERM.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from architecture_zoo_v15 import Network as LatentNetwork, clean

MODES = ("shared_global", "local_gate")


class Network(LatentNetwork):
    def __init__(self, mode: str, centers, target_median: float, target_scale: float):
        if mode not in MODES:
            raise ValueError(f"Unknown v17 mode: {mode}")
        super().__init__("latent", centers, target_median, target_scale)
        self.mode = mode

    def features(self, x, relative, valid):
        inp = clean(torch.cat((x.float(), relative[..., None].float()), dim=-1), valid)
        stem = clean(F.gelu(self.stem(inp)), valid)
        before_memory = self.blocks[0](stem, valid)
        # Compute both paths in either mode. Same local block parameters; no
        # added capacity. Backward paths intentionally depend on readout routing.
        local = self.blocks[2](before_memory, valid)
        global_features = self.blocks[2](self.blocks[1](before_memory, valid), valid)
        return local, global_features

    def forward(self, x, relative, valid, absolute_depth=None):
        local, global_features = self.features(x, relative, valid)
        local_output = clean(self.output(self.norm(local)).float(), valid)
        global_output = clean(self.output(self.norm(global_features)).float(), valid)
        gate = local_output[..., :1] if self.mode == "local_gate" else global_output[..., :1]
        return gate, global_output[..., 1:]

    @torch.no_grad()
    def mechanism_statistics(self):
        return {**super().mechanism_statistics(), "default_head_reads": self.mode,
                "density_head_reads": "global_memory", "local_blocks_shared_within_SW": True}

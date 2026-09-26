"""Replace learned sigmoid gates with constant 0.5 and remove unused gate projections."""
import torch
from torch import nn
from torch.nn import functional as F
from .blur_no_point_sequence import Network as Baseline
from .multiscale_sequence import GatedBlock

class UngatedBlock(nn.Module):

    def __init__(self, old):
        super().__init__()
        self.norm = old.norm
        self.depthwise = old.depthwise
        self.project = old.project
        self.dropout = old.dropout
        self.gain = old.gain
        c = old.expand.in_channels
        with torch.random.fork_rng(devices=[]):
            self.expand = nn.Conv1d(c, c, 1)
        with torch.no_grad():
            self.expand.weight.copy_(old.expand.weight[:c])
            self.expand.bias.copy_(old.expand.bias[:c])

    def forward(self, x):
        return x + self.gain * self.project(self.dropout(0.5 * F.gelu(self.expand(self.depthwise(self.norm(x))))))

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        count = 0
        for parent in list(self.modules()):
            for (name, child) in list(parent.named_children()):
                if isinstance(child, GatedBlock):
                    setattr(parent, name, UngatedBlock(child))
                    count += 1
        assert count == 9

"""Remove the point MLP bypass; retain all sequence inputs and original decoder."""
import torch
from torch import nn
from .blur_no_dropout_sequence import Network as Baseline

class EmptyPoint(nn.Module):

    def forward(self, x):
        return x[:, :0, :]

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        self.point = EmptyPoint()
        old = self.combine[0]
        with torch.random.fork_rng(devices=[]):
            new = nn.Conv1d(64, 128, 1)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, :64])
            new.bias.copy_(old.bias)
        self.combine[0] = new

"""Remove absolute-depth stem input, preserving relative depth and all measured features."""
import torch
from torch import nn
from .blur_ungated_sequence import Network as Baseline

class RelativeOnlyStem(nn.Conv1d):

    def forward(self, x):
        return super().forward(x[:, :-1, :])

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        old = self.stem
        with torch.random.fork_rng(devices=[]):
            new = RelativeOnlyStem(features + 2, 64, 1)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, :-1])
            new.bias.copy_(old.bias)
        self.stem = new

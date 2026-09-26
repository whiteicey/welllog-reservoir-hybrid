"""Disable existing dropout on the verified three-stage blur model."""
import torch
from torch import nn
from .blur_down_sequence import Network as Baseline

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

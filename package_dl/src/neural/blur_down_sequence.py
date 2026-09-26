"""Fixed low-pass filter before each stride-two convolution."""
import torch
from torch import nn
from torch.nn import functional as F
from .depth_norm_joint_sequence import Network as Baseline

class BlurDown(nn.Module):

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.register_buffer('kernel', torch.tensor([0.25, 0.5, 0.25]).reshape(1, 1, 3))

    def smooth(self, x):
        return F.conv1d(F.pad(x, (1, 1), mode='replicate'), self.kernel.to(x.dtype).expand(x.shape[1], 1, 3), groups=x.shape[1])

    def forward(self, x):
        return self.conv(self.smooth(x))

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for name in ['down1', 'down2', 'down3']:
            setattr(self, name, BlurDown(getattr(self, name)))

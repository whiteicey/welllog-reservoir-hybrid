"""Learn expected SW utility of retain-original and fixed legal actions."""
import torch
from torch import nn


class ActionPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(212,64),nn.GELU(),nn.Linear(64,66))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-2.)
        with torch.no_grad():self.net[-1].bias[0]=2.
        self.register_buffer('candidates',torch.cat((torch.logspace(torch.log10(torch.tensor(8.305)),torch.log10(torch.tensor(99.9)),64),torch.tensor([99.9]))))

    def forward(self,features):return self.net(features)

    def actions(self,original):
        return torch.cat((original[:,2:3],self.candidates[None].expand(len(original),-1)),1)

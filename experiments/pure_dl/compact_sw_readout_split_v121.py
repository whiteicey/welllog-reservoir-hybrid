"""Matched SW final residual branches: shared versus gate/density-specific features."""
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Parent,load_data
STUDY=ROOT/'results/deep_learning/compact_sw_readout_split_v121_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_readout_split_v121_fold5'
STAGES={n:('SW',1024,True,1.,3e-4) for n in ('shared','split')}
class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128);self.reduce=nn.Linear(128,64);self.expand=nn.Linear(64,128)
        nn.init.zeros_(self.expand.weight);nn.init.zeros_(self.expand.bias)
    def forward(self,h):return .25*torch.tanh(self.expand(F.gelu(self.reduce(self.norm(h)))).float()/.25)
class Readout(nn.Module):
    def __init__(self,original,mode):
        super().__init__();self.weight=original.weight;self.bias=original.bias;self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261121);self.branches=nn.ModuleList([Residual(),Residual()])
    def forward(self,h):
        a,b=[branch(h) for branch in self.branches]
        if self.mode=='shared':a=b=(a+b)*.5
        gate=F.linear((h+a.to(h.dtype)),self.weight[:1],self.bias[:1])
        density=F.linear((h+b.to(h.dtype)),self.weight[1:],self.bias[1:])
        return torch.cat((gate,density),-1)
class Network(Parent):
    def __init__(self,metadata,mode):
        super().__init__(metadata,'local');self.readout_mode=mode;self.output=Readout(self.output,mode)
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(readout_mode=self.readout_mode,matched_two_residual_branches=True,
            residual_width=64,residual_output_width=128,extra_parameters=33664,zero_residual_initial=True,
            shared_uses_branch_mean_for_both_outputs=True,split_uses_branch0_gate_branch1_density=True,
            original_SW_initial_function=True,parameter_capacity_equal=True,backbone_still_shared=True)
        return info
def make_model(stage,master,metadata):return Network(metadata,stage)

"""One input-conditional POR family gate; shared eight component locations/scales."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline


def expand_families(raw,family_logit):
    assert raw.shape[-3:]==(3,3,8) and family_logit.shape==raw.shape[:-3]+(1,)
    logweights=raw[...,0,:].float().log_softmax(-1)
    fixed=torch.full_like(family_logit.float(),-math.log(2.))
    first=torch.cat([F.logsigmoid(family_logit.float()),fixed,fixed],-1)
    second=torch.cat([F.logsigmoid(-family_logit.float()),fixed,fixed],-1)
    weights=torch.cat([logweights+first[...,None],logweights+second[...,None]],-1)
    means=torch.cat([raw[...,1,:],raw[...,1,:]],-1)
    scales=torch.cat([raw[...,2,:],raw[...,2,:]],-1)
    return torch.stack([weights,means,scales],-2)


class FamilyHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.gate=old.gate;self.density=old.density
        self.family_gate=nn.Conv1d(old.gate.in_channels,1,1)
        nn.init.zeros_(self.family_gate.weight);nn.init.zeros_(self.family_gate.bias)
    def forward(self,x):
        raw=self.density(x).transpose(1,2).reshape(len(x),-1,3,3,8)
        family=self.family_gate(x).transpose(1,2)
        return self.gate(x).transpose(1,2),expand_families(raw,family)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):
            for name in ['main_head','aux_mid','aux_coarse']:
                setattr(self,name,FamilyHead(getattr(self,name)))

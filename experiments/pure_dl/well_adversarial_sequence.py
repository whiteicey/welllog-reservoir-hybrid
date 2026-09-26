"""Training-only well classifier with gradient reversal; inference is unchanged."""
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,coefficient):
        ctx.coefficient=float(coefficient)
        return value.view_as(value)
    @staticmethod
    def backward(ctx,gradient):
        return -ctx.coefficient*gradient,None


def grl_coefficient(epoch_zero_based):
    # Epochs1..10:0;11..20:.001..01; thereafter.01, fixed before scoring.
    return .01*max(0.,min(1.,(int(epoch_zero_based)-9)/10.))


def make_domain_mapping(training_wells,total_wells=80):
    ids=sorted(map(int,training_wells))
    assert len(ids)==64 and len(set(ids))==64 and min(ids)>=0 and max(ids)<total_wells
    mapping=torch.full((total_wells,),-1,dtype=torch.long)
    mapping[ids]=torch.arange(64)
    return mapping


def domain_labels(well_ids,mapping):
    assert ((well_ids>=0)&(well_ids<len(mapping))).all()
    labels=mapping[well_ids]
    assert (labels>=0).all(),'Validation or unknown wells cannot enter domain training'
    return labels


def pool_core(fine,valid):
    assert fine.ndim==3 and fine.shape[1]==128 and fine.shape[-1]==256
    assert valid.shape==(len(fine),256)
    mask=valid.clone();mask[:,:64]=False;mask[:,192:]=False
    counts=mask.sum(-1)
    pooled=fine.float().masked_fill(~mask[:,None],0).sum(-1)/counts.clamp_min(1)[:,None]
    return pooled,counts>0


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.domain_head=nn.Linear(128,64)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None,
                return_domain=False,domain_coefficient=0.):
        if not return_domain:
            return super().forward(features,relative_depth,valid,auxiliary,absolute_depth=absolute_depth)
        assert self.training,'Domain branch is training-only'
        captured=[]
        handle=self.combine.register_forward_hook(lambda module,args,output:captured.append(output))
        try:
            outputs=super().forward(features,relative_depth,valid,auxiliary,absolute_depth=absolute_depth)
        finally:
            handle.remove()
        assert len(captured)==1
        pooled,usable=pool_core(captured[0],valid)
        logits=self.domain_head(GradientReverse.apply(pooled,domain_coefficient))
        return outputs,logits,usable

    def backbone_parameters(self):
        return [parameter for name,parameter in self.named_parameters() if not name.startswith('domain_head.')]

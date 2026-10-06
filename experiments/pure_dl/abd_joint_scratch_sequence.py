"""Three neural sources trained jointly from their original per-role seeds."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Dual, retained_indices, VIEW_NAMES, calibration_indices
from paired_detached_router_sequence import routed_merge
from global_cosine_sequence import Network as Single
from por_t3_patch_loss import head_loss


def equal_merge(outputs):
    gates=torch.stack([g.float() for g,_ in outputs])
    log_default=F.logsigmoid(gates).logsumexp(0)
    survival_each=F.logsigmoid(-gates)
    survival=survival_each.logsumexp(0)
    gate=log_default-survival
    shares=survival_each-survival[None]
    weights=torch.cat([r[...,0,:].float().log_softmax(-1)+shares[i,...,None] for i,(_,r) in enumerate(outputs)],-1)
    means=torch.cat([r[...,1,:].float() for _,r in outputs],-1)
    scales=torch.cat([r[...,2,:].float() for _,r in outputs],-1)
    return gate,torch.stack([weights,means,scales],-2)


class Network(nn.Module):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__();models=[]
        # CPU-only generator reseeding preserves the outer CUDA minibatch RNG.
        for i,seed in enumerate([20261006,20261008,20261006]):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+4)
                model=(Dual(features,initial_means,retain_indices=retain_indices,numeric_indices=numeric_indices,
                            conditioning_indices=conditioning_indices) if i<2 else Single(features,initial_means))
            models.append(model)
        self.members=nn.ModuleList(models)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        if not auxiliary:
            return equal_merge([m(features,relative_depth,valid,absolute_depth=absolute_depth) for m in self.members])
        mains=[];aux=[]
        for i,model in enumerate(self.members):
            if i<2:
                outputs=[m(features,relative_depth,valid,True,absolute_depth) for m in model.members]
                branch_main=[out[0] for out in outputs]
                logits=model.route_logits(features,valid,branch_main)
                # Unlike the original detached router-only loss, fused loss reaches every backbone.
                mains.append(routed_merge(branch_main,logits));aux.append(outputs)
            else:
                output=model(features,relative_depth,valid,True,absolute_depth)
                mains.append(output[0]);aux.append([output])
        return equal_merge(mains),aux


def objective(outputs,target,default,mask,deep_supervision=False,weights=None,*,epoch):
    assert deep_supervision and weights is not None
    main,aux=outputs
    loss=head_loss(main,target,default,mask,weights)
    factor=max(0.,1.-epoch/40.)
    if factor:
        for source in aux:
            for output in source:
                loss=loss+factor/(3*len(source))*(.2*head_loss(output[1],target,default,mask,weights)+.1*head_loss(output[2],target,default,mask,weights))
    return loss

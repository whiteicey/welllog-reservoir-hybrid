"""Matched discrete utility-kernel score with self-interaction versus expected utility."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_sam_utility_v171 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator
from perm_ranked_probability_v154 import likelihood
STUDY=ROOT/'results/deep_learning/perm_kernel_score_v188_20261001'
OUT=ROOT/'outputs/pure_dl_perm_kernel_score_v188_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        support=torch.cat((torch.tensor([.01]),self.physical_centers.detach().cpu())).clamp_min(1e-12).log10()
        self.register_buffer('utility_kernel',(1-(support[:,None]-support[None,:]).abs()).clamp_min(0))
        self.register_buffer('kernel_support_log10',support)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(joint_ranked_probability_loss=False,conditional_continuous_RPS=True,
                    continuous_labels_only_for_RPS=True,RPS_default_rows_zero=True,
                    RPS_global_row_normalization_unchanged=True,no_trained_v171_weights_loaded=True,
                    train_only_expected_logratio_utility_weight=.1,utility_kernel_self_interaction_weight=.5,
                    discrete_kernel_score=True,continuous_propriety_not_claimed=True,
                    no_trained_v174_weights_loaded=True)
        return info
def conditional_ranked_loss(r,y,d,edges):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf(delta/(sd*math.sqrt(2))))
    conditional=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    result=(conditional-(y[:,None]<=edges[None,:]).float()).square().mean(-1)
    return result.masked_fill(d,0)
def utility_loss(g,r,y,d,edges,centers,median,scale,kernel,support):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf(delta/(sd*math.sqrt(2))))
    mixed=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    full=torch.cat((torch.zeros_like(mixed[:,:1]),mixed,torch.ones_like(mixed[:,:1])),-1)
    mass=full.diff(dim=-1).clamp_min(0);mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    gate=g.sigmoid();p=torch.cat((gate[:,None],(1-gate[:,None])*mass),-1)
    target_log=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    ky=(1-(support[None,:]-target_log[:,None]).abs()).clamp_min(0)
    attraction=(p*ky).sum(-1)
    self_interaction=((p@kernel)*p).sum(-1)
    # Half the squared kernel distance to a Dirac label. Attraction gradient
    # matches v174; only .5*p^T*K*p is added (constant shift is irrelevant).
    return .5-attraction+.5*self_interaction

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=likelihood(output,truth,default,mask,quality,1,mode)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float();rps=[];utility=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        rps.append(checkpoint(conditional_ranked_loss,r[take],y[take],d[take],model.rps_edges,use_reentrant=False,preserve_rng_state=False))
        utility.append(checkpoint(utility_loss,g[take],r[take],y[take],d[take],model.rps_edges,model.physical_centers,model.target_median,model.target_scale,model.utility_kernel,model.kernel_support_log10,use_reentrant=False,preserve_rng_state=False))
    extra=(q*(torch.cat(rps)+.1*torch.cat(utility))).mean()/3
    return original[0]+extra,original[1],original[2]+3*extra
def make_model(stage,master,metadata):return Network(master,stage)

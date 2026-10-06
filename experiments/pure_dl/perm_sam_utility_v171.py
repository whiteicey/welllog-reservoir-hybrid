"""Matched-capacity positive RPS readout plus train-only PERM log-ratio utility."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective as proper_objective
STUDY=ROOT/'results/deep_learning/perm_sam_utility_v171_20261001'
OUT=ROOT/'outputs/pure_dl_perm_sam_utility_v171_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][1],dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][1])))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][1])))
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(train_only_expected_logratio_utility_weight=.1,same_capacity_as_v154=True,
                    own_PERM_labels_only=True,trained_v154_head_not_loaded=True,
                    utility_proxy_not_exact_decoded_argmax=True,original_decoder_unchanged=True)
        return info

def utility_loss(g,r,y,d,edges,centers,median,scale):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf(delta/(sd*math.sqrt(2))))
    mixed=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    full=torch.cat((torch.zeros_like(mixed[:,:1]),mixed,torch.ones_like(mixed[:,:1])),-1)
    mass=full.diff(dim=-1).clamp_min(0);mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    # Canonical PERM utility is triangular in log10(pred/true); default is0.01.
    target_log=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    utility=(1-(centers.clamp_min(1e-12).log10()[None,:]-target_log[:,None]).abs()).clamp_min(0)
    atom_utility=(1-(-2.-target_log).abs()).clamp_min(0)
    gate=g.sigmoid()
    return 1-gate*atom_utility-(1-gate)*(mass*utility).sum(-1)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=proper_objective(output,truth,default,mask,quality,1,mode,model)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float();parts=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        parts.append(checkpoint(utility_loss,g[take],r[take],y[take],d[take],model.rps_edges,
                     model.physical_centers,model.target_median,model.target_scale,
                     use_reentrant=False,preserve_rng_state=False))
    loss=(q*torch.cat(parts)).mean()/3
    return original[0]+.1*loss,original[1],original[2]+.3*loss

def make_model(stage,master,metadata):return Network(master,stage)

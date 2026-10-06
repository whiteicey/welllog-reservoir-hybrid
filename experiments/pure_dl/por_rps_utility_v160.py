"""Matched-capacity positive RPS readout plus train-only POR tolerance utility."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from por_ranked_probability_v151 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective as proper_objective
from compact_por_mixed_family_v113 import choose
from compact_por_cauchy_v112 import IQR_SCALE
STUDY=ROOT/'results/deep_learning/por_rps_utility_v160_20261001'
OUT=ROOT/'outputs/pure_dl_por_rps_utility_v160_fold5'
STAGES={'joint_rps':('POR',256,False,0.,3e-4)}

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][0],dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][0])))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][0])))
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(train_only_expected_tolerance_utility_weight=.1,same_capacity_as_v151=True,
                    own_POR_labels_only=True,trained_v151_head_not_loaded=True,
                    utility_proxy_not_exact_decoded_argmax=True,original_decoder_unchanged=True)
        return info

def utility_loss(g,r,y,d,edges,centers,median,scale):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    phi=torch.atan(delta/sd)
    cdf=choose('t3_cauchy',.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi,
               .5+torch.atan(delta/(sd*IQR_SCALE))/math.pi).clamp(0,1)
    mixed=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    full=torch.cat((torch.zeros_like(mixed[:,:1]),mixed,torch.ones_like(mixed[:,:1])),-1)
    mass=full.diff(dim=-1).clamp_min(0);mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    physical=torch.where(d,torch.full_like(y,.1),torch.expm1(y*scale+median))
    tolerance=.08*(physical.abs()+1e-5)
    utility=(1-(centers[None,:]-physical[:,None]).abs()/tolerance[:,None]).clamp_min(0)
    atom_utility=(1-(physical-.1).abs()/tolerance).clamp_min(0)
    gate=g.sigmoid()
    return 1-gate*atom_utility-(1-gate)*(mass*utility).sum(-1)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==0
    original=proper_objective(output,truth,default,mask,quality,0,mode,model)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,0].float();d=default[mask][:,0];q=quality[mask].detach().float();parts=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        parts.append(checkpoint(utility_loss,g[take],r[take],y[take],d[take],model.rps_edges,
                     model.physical_centers,model.target_median,model.target_scale,
                     use_reentrant=False,preserve_rng_state=False))
    loss=(q*torch.cat(parts)).mean()/3
    return original[0]+.1*loss,original[1],original[2]+.3*loss

def make_model(stage,master,metadata):return Network(master,stage)

"""Transfer evidence-positive joint RPS supervision to the effective PERM readout."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_joint_density_v136 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective as likelihood
from src.utils import DEFAULTS
STUDY=ROOT/'results/deep_learning/perm_ranked_probability_v156_20261001'
OUT=ROOT/'outputs/pure_dl_perm_ranked_probability_v156_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][1][1:-1],dtype=torch.float32))
        value=(math.log10(DEFAULTS['PERM'])-float(master['regression_median'][1]))/float(master['regression_scale'][1])
        self.register_buffer('default_z',torch.tensor(value,dtype=torch.float32))
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(joint_ranked_probability_loss=True,rps_coefficient=2.,original_finite_edges=255,
            own_PERM_default_and_continuous_labels=True,same_capacity_as_v136=True,
            original_transform='log10',default_physical=DEFAULTS['PERM'],trained_v136_weights_not_loaded=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)

def ranked_loss(g,r,y,d,edges,default_z):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf(delta/(sd*math.sqrt(2))))
    conditional=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    gate=g.sigmoid()[:,None];joint=gate*(default_z<=edges).float()[None,:]+(1-gate)*conditional
    target=torch.where(d,default_z,y)
    return (joint-(target[:,None]<=edges[None,:]).float()).square().mean(-1)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=likelihood(output,truth,default,mask,quality,1,mode)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float();parts=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        parts.append(checkpoint(ranked_loss,g[take],r[take],y[take],d[take],model.rps_edges,model.default_z,
            use_reentrant=False,preserve_rng_state=False))
    rps=2*(q*torch.cat(parts)).mean()/3
    return original[0]+rps,original[1],original[2]+3*rps

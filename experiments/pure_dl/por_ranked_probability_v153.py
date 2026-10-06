"""Joint atom/continuous ranked probability supervision on original finite edges."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from por_joint_density_width_v140 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator
from compact_por_mixed_family_v113 import choose,channel_objective as likelihood
from compact_por_cauchy_v112 import IQR_SCALE
STUDY=ROOT/'results/deep_learning/por_ranked_probability_v153_20261001'
OUT=ROOT/'outputs/pure_dl_por_ranked_probability_v153_fold5'
STAGES={'joint_rps':('POR',256,False,0.,3e-4)}

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][0][1:-1],dtype=torch.float32))
        self.register_buffer('default_z',torch.tensor((math.log1p(.1)-float(master['regression_median'][0]))/float(master['regression_scale'][0]),dtype=torch.float32))
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(joint_ranked_probability_loss=True,rps_coefficient=2.,rps_finite_original_edges=255,
            true_default_labels_supervise_conditional_distribution=True,additional_trainable_parameters=0,
            no_continuous_infinite_support_CRPS_claim=True,no_action_utility_or_softargmax_loss=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)

def ranked_loss(g,r,y,d,edges,default_z):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    phi=torch.atan(delta/sd)
    cdf=choose('t3_cauchy',.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi,
        .5+torch.atan(delta/(sd*IQR_SCALE))/math.pi).clamp(0,1)
    conditional=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    gate=g.sigmoid()[:,None];atom=(default_z<=edges).float()[None,:]
    joint=gate*atom+(1-gate)*conditional
    target=torch.where(d,default_z,y)
    empirical=(target[:,None]<=edges[None,:]).float()
    return (joint-empirical).square().mean(-1)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    original=likelihood(output,truth,default,mask,quality,0,'t3_cauchy')
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,0].float();d=default[mask][:,0];q=quality[mask].detach().float()
    losses=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        losses.append(checkpoint(ranked_loss,g[take],r[take],y[take],d[take],model.rps_edges,model.default_z,
            use_reentrant=False,preserve_rng_state=False))
    rps=2*(q*torch.cat(losses)).mean()/3
    return original[0]+rps,original[1],original[2]+3*rps

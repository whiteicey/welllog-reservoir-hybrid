"""Conditional continuous CDF supervision on the positive SAM/utility foundation."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_sam_utility_v171 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,utility_loss
from perm_ranked_probability_v154 import likelihood
STUDY=ROOT/'results/deep_learning/perm_conditional_cdf_v174_20261001'
OUT=ROOT/'outputs/pure_dl_perm_conditional_cdf_v174_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(joint_ranked_probability_loss=False,conditional_continuous_RPS=True,
                    continuous_labels_only_for_RPS=True,RPS_default_rows_zero=True,
                    RPS_global_row_normalization_unchanged=True,no_trained_v171_weights_loaded=True)
        return info
def conditional_ranked_loss(r,y,d,edges):
    delta=edges[None,:,None]-r[:,1,None,:];sd=(F.softplus(r[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf(delta/(sd*math.sqrt(2))))
    conditional=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    result=(conditional-(y[:,None]<=edges[None,:]).float()).square().mean(-1)
    return result.masked_fill(d,0)
def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=likelihood(output,truth,default,mask,quality,1,mode)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float();rps=[];utility=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        rps.append(checkpoint(conditional_ranked_loss,r[take],y[take],d[take],model.rps_edges,use_reentrant=False,preserve_rng_state=False))
        utility.append(checkpoint(utility_loss,g[take],r[take],y[take],d[take],model.rps_edges,model.physical_centers,model.target_median,model.target_scale,use_reentrant=False,preserve_rng_state=False))
    extra=(q*(torch.cat(rps)+.1*torch.cat(utility))).mean()/3
    return original[0]+extra,original[1],original[2]+3*extra
def make_model(stage,master,metadata):return Network(master,stage)

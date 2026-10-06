"""Retain effective Gaussian joint-RPS readout; add only analytic CRPS0.1."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective as positive_objective
STUDY=ROOT/'results/deep_learning/perm_additive_crps_v195_20261001'
OUT=ROOT/'outputs/pure_dl_perm_additive_crps_v195_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(additive_conditional_analytic_CRPS_weight=.1,existing_NLL_BCE_jointRPS_unchanged=True,
                    original_Gaussian_location_logweight_structure_retained=True,no_new_parameters=True,
                    continuous_CRPS_default_rows_zero=True,CRPS_global_row_normalization_unchanged=True,
                    no_trained_v154_head_loaded=True,matched80_14_control='v165 adamw')
        return info

def expected_absolute(delta,sd):
    t=delta/sd
    return sd*math.sqrt(2/math.pi)*torch.exp(-.5*t.square())+delta*torch.erf(t/math.sqrt(2))

def conditional_crps(r,y,d):
    weight=r[:,0].softmax(-1);mu=r[:,1];sd=F.softplus(r[:,2])+.025
    attraction=(weight*expected_absolute(y[:,None]-mu,sd)).sum(-1)
    pair_delta=mu[:,:,None]-mu[:,None,:]
    pair_sd=(sd[:,:,None].square()+sd[:,None,:].square()).sqrt()
    self_energy=(weight[:,:,None]*weight[:,None,:]*expected_absolute(pair_delta,pair_sd)).sum((-2,-1))
    return (attraction-.5*self_energy).masked_fill(d,0)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=positive_objective(output,truth,default,mask,quality,1,mode,model)
    _,raw=output;r=raw[mask][:,0].float();y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float()
    parts=[]
    for start in range(0,len(r),1024):
        take=slice(start,start+1024)
        parts.append(checkpoint(conditional_crps,r[take],y[take],d[take],use_reentrant=False,preserve_rng_state=False))
    extra=(q*torch.cat(parts)).mean()/3
    return original[0]+.1*extra,original[1],original[2]+.3*extra

def make_model(stage,master,metadata):return Network(master,stage)

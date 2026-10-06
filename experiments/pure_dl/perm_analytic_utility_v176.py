"""Analytic Gaussian expected PERM utility, replacing the finite-bin training proxy."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_conditional_cdf_v174 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,conditional_ranked_loss
from perm_ranked_probability_v154 import likelihood
STUDY=ROOT/'results/deep_learning/perm_analytic_utility_v176_20261001'
OUT=ROOT/'outputs/pure_dl_perm_analytic_utility_v176_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(analytic_Gaussian_expected_logratio_utility=True,finite_bin_training_utility_replaced=True,
                    original_inference_mass_grid_unchanged=True,no_trained_v174_weights_loaded=True,
                    analytic_utility_is_not_exact_decoded_argmax=True)
        return info
def analytic_utility_loss(g,r,y,d,median,scale):
    # Let Z=log10(pred/target)~N(m,sigma). Integrate(1-|Z|)+ on[-1,0] and[0,1].
    target=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    m=r[:,1]*scale+median-target[:,None];sigma=(F.softplus(r[:,2])+.025)*scale
    a=(-1-m)/sigma;b=-m/sigma;c=(1-m)/sigma
    phi=lambda z:torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    cdf=lambda z:.5*(1+torch.erf(z/math.sqrt(2)))
    left=cdf(b)-cdf(a);right=cdf(c)-cdf(b)
    component=((1+m)*left+(1-m)*right+sigma*(phi(a)-2*phi(b)+phi(c))).clamp(0,1)
    continuous=(r[:,0].softmax(-1)*component).sum(-1)
    atom=(1-(-2.-target).abs()).clamp_min(0);gate=g.sigmoid()
    return 1-gate*atom-(1-gate)*continuous
def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    original=likelihood(output,truth,default,mask,quality,1,mode)
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    y=truth[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float();rps=[];utility=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        rps.append(checkpoint(conditional_ranked_loss,r[take],y[take],d[take],model.rps_edges,use_reentrant=False,preserve_rng_state=False))
        utility.append(checkpoint(analytic_utility_loss,g[take],r[take],y[take],d[take],model.target_median,model.target_scale,use_reentrant=False,preserve_rng_state=False))
    extra=(q*(torch.cat(rps)+.1*torch.cat(utility))).mean()/3
    return original[0]+extra,original[1],original[2]+3*extra
def make_model(stage,master,metadata):return Network(master,stage)

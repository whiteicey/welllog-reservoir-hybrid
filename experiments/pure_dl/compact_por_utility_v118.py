"""Proper mixed-family POR NLL plus original-grid soft decision utility."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from compact_por_scale_v114 import ROOT,SOURCE,PARENT,Network as Parent,Evaluator,load_data,channel_objective
from compact_por_mixed_family_v113 import choose
from compact_por_cauchy_v112 import IQR_SCALE
STUDY=ROOT/'results/deep_learning/compact_por_utility_v118_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_utility_v118_fold5'
STAGES={n:('POR',256,False,a,3e-5) for n,a in (('alpha025',.25),('alpha050',.5))}
class Network(Parent):
    def __init__(self,master,mode):super().__init__(master,'location_only');self.utility_mode=mode
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(mode=self.utility_mode,utility_alpha=STAGES[self.utility_mode][3],utility_temperature=.05,original_action_grid_and_true_kernel=True,historical_likelihood_control='v114 location_only',same_initial_function=True);return info
def make_model(stage,master,metadata):return Network(master,stage)
def differentiable_probability(weight,mu,sd,edges):
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges));delta=safe[None,:,None]-mu[:,None,:];ref=sd[:,None,:]
    phi=torch.atan(delta/ref);t3=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cauchy=.5+torch.atan(delta/(ref*IQR_SCALE))/math.pi
    cdf=choose('t3_cauchy',t3,cauchy)
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1).clamp_min(0)*weight[:,None,:]).sum(-1)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
class DecisionObjective(nn.Module):
    def __init__(self,evaluator,master,mode):
        super().__init__();self.alpha=STAGES[mode][3];self.temperature=.05
        self.register_buffer('actions',torch.as_tensor(evaluator.actions,dtype=torch.float32));self.register_buffer('utilities',evaluator.utilities.detach().clone());self.register_buffer('edges',evaluator.edges.detach().clone())
        self.target_median=float(master['regression_median'][0]);self.target_scale=float(master['regression_scale'][0])
    def forward(self,output,labels,default,mask,quality):
        likelihood=channel_objective(output,labels,default,mask,quality,0,'location_only')[0]
        gate,raw=output;g=gate[mask][:,0].float().sigmoid();r=raw[mask][:,0].float()
        d=default[mask][:,0];z=labels[mask][:,0].float();q=quality[mask].detach().float()
        conditional=differentiable_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g[:,None],(1-g[:,None])*conditional),-1)
        action_probability=((joint@self.utilities)/self.temperature).softmax(-1)
        physical_y=torch.expm1(z*self.target_scale+self.target_median);physical_y=torch.where(d,torch.full_like(physical_y,.1),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.08*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        decision=(q*(1-(action_probability*true_utility).sum(-1))).mean()/3
        return likelihood+self.alpha*decision,likelihood,decision

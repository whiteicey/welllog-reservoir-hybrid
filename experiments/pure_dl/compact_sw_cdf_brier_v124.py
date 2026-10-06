"""Own SW finite-threshold CDF Brier calibration on unchanged Gaussian local model."""
import math
import torch
from compact_frequency_v104 import ROOT,SOURCE,Network as Parent,load_data
from sw_head_utility_v80r1 import DecisionObjective as BaseObjective,differentiable_gaussian_probability
from train_scaling_v21 import channel_objective
STUDY=ROOT/'results/deep_learning/compact_sw_cdf_brier_v124_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_cdf_brier_v124_fold5'
CDF_WEIGHT={'cdf1':1.,'cdf4':4.}
STAGES={n:('SW',1024,True,1.,3e-4) for n in CDF_WEIGHT}
class Network(Parent):
    def __init__(self,metadata,mode):super().__init__(metadata,'local');self.cdf_mode=mode
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(cdf_weight=CDF_WEIGHT[self.cdf_mode],finite_threshold_CDF_Brier=True,
            original_NLL_and_utility1_retained=True,decoder_unchanged=True,network_capacity_same=True)
        return info
def make_model(stage,master,metadata):return Network(metadata,stage)
class DecisionObjective(BaseObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'joint');self.alpha=1.;self.cdf_weight=CDF_WEIGHT[mode]
        self.default_z=(math.log(99.9)-self.target_median)/self.target_scale
    def forward(self,output,labels,default,mask,quality):
        likelihood=channel_objective(output,labels,default,mask,quality,2)[0]
        gate,raw=output;g=gate[mask][:,0].float().sigmoid();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        conditional=differentiable_gaussian_probability(r[:,0].softmax(-1),r[:,1],torch.nn.functional.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g[:,None],(1-g[:,None])*conditional),-1)
        policy=((joint@self.utilities)/self.temperature).softmax(-1)
        physical_y=torch.exp(z*self.target_scale+self.target_median);physical_y=torch.where(d,torch.full_like(physical_y,99.9),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        decision=(q*(1-(policy*true_utility).sum(-1))).mean()/3
        thresholds=self.edges[1:-1];default_cdf=(self.default_z<=thresholds).float()
        predicted_cdf=g[:,None]*default_cdf[None]+(1-g[:,None])*conditional.cumsum(-1)[:,:-1]
        true_z=torch.where(d,torch.full_like(z,self.default_z),z)
        truth_cdf=(true_z[:,None]<=thresholds[None]).float()
        brier=(q*(predicted_cdf-truth_cdf).square().mean(-1)).mean()/3
        return likelihood+decision+self.cdf_weight*brier,likelihood,decision

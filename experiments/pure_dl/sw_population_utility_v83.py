"""Train-only population weighting of the same full-network SW joint utility."""
import numpy as np
import torch
from sw_utility_strength_v82 import ROOT,Network as Parent,Evaluator,SOURCE,load_data as parent_data
from sw_head_utility_v80r1 import DecisionObjective as ParentObjective,differentiable_gaussian_probability
from train_scaling_v21 import channel_objective

STUDY=ROOT/'results/deep_learning/sw_population_utility_v83_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_utility_strength_v82_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5'
MODES={mode:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=1.,numeric_share=share)
       for mode,share in (('balanced',.5),('numeric75',.75))}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def load_data():
    data,meta=parent_data()
    defaults=data['default'][data['train_raw'],2]
    n=int(defaults.numel());nd=int(defaults.sum().item());nn=n-nd
    meta['utility_population']=dict(training_rows=n,default_rows=nd,numeric_rows=nn,
        default_fraction=nd/n,numeric_fraction=nn/n,
        fit_scope='original64 training raw SW labels only; unweighted counts; original Q50 applied separately',
        validation_population_used=False)
    return data,meta

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('a100',metadata);self.study_mode=mode
        self.parent_info.update(utility_population_share=MODES[mode]['numeric_share'],
            historical_population_reference=str(CONTROL.relative_to(ROOT)),loaded_v82_trained_weights=False,
            only_hyperparameter_changed='joint utility default/numeric population allocation; BCE/NLL/Q/alpha1 unchanged; same original v56 body/reset prior')

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'joint');self.alpha=1.
        share=MODES[mode]['numeric_share'];pop=metadata['utility_population']
        self.default_weight=(1-share)/pop['default_fraction']
        self.numeric_weight=share/pop['numeric_fraction']
    def forward(self,output,labels,default,mask,quality):
        likelihood,_,_=channel_objective(output,labels,default,mask,quality,2)
        gate,raw=output;g=gate[mask][:,0].float().sigmoid();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        conditional=differentiable_gaussian_probability(r[:,0].softmax(-1),r[:,1],torch.nn.functional.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g[:,None],(1-g[:,None])*conditional),-1)
        predicted_utility=joint@self.utilities
        action_probability=(predicted_utility/self.temperature).softmax(-1)
        physical_y=torch.exp(z*self.target_scale+self.target_median)
        physical_y=torch.where(d,torch.full_like(physical_y,99.9),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        reward=(action_probability*true_utility).sum(-1)
        population_weight=torch.where(d,self.default_weight,self.numeric_weight)
        decision=(population_weight*q*(1-reward)).mean()/3
        return likelihood+self.alpha*decision,likelihood,decision

"""Same random local SW; utility surrogate with two-level action probabilities."""
import torch
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_frequency_v104 import Network as LocalNetwork
from sw_head_utility_v80r1 import DecisionObjective as FlatObjective,differentiable_gaussian_probability
from train_scaling_v21 import channel_objective
import train_invariant as base

STUDY=ROOT/'results/deep_learning/compact_hierarchical_utility_v110_20261001'
OUT=ROOT/'outputs/pure_dl_compact_hierarchical_utility_v110_fold5'
STAGES={'hierarchical':('SW',1024,True,1.,3e-4)}

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return LocalNetwork(metadata,'local')

class DecisionObjective(FlatObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'joint');self.alpha=1.
        default_action=self.actions==base.DEFAULTS['SW']
        assert int(default_action.sum())==1
        self.default_action_index=int(default_action.nonzero()[0,0])
        self.register_buffer('continuous_indices',(~default_action).nonzero().flatten())

    def forward(self,output,labels,default,mask,quality):
        likelihood,_,_=channel_objective(output,labels,default,mask,quality,2)
        gate,raw=output;g=gate[mask][:,0].float().sigmoid();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        conditional=differentiable_gaussian_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g[:,None],(1-g[:,None])*conditional),-1)
        expected=joint@self.utilities
        continuous=expected.index_select(1,self.continuous_indices)
        within=(continuous/self.temperature).softmax(-1)
        # Compare maxima between the two groups, without a count-dependent
        # logsumexp bonus. All rewards still come from the original numeric kernel.
        group_scores=torch.stack((expected[:,self.default_action_index],continuous.amax(-1)),-1)
        groups=(group_scores/self.temperature).softmax(-1)
        physical_y=torch.exp(z*self.target_scale+self.target_median)
        physical_y=torch.where(d,torch.full_like(physical_y,base.DEFAULTS['SW']),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        reward=groups[:,0]*true_utility[:,self.default_action_index]+groups[:,1]*(within*true_utility.index_select(1,self.continuous_indices)).sum(-1)
        decision=(q*(1-reward)).mean()/3
        return likelihood+self.alpha*decision,likelihood,decision

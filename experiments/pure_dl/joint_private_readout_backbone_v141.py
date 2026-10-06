"""Random joint student with three private nonlinear predictive readouts."""
import torch
from torch import nn
from joint_representation_backbone_v132r1 import ROOT,SOURCE,Network as Aligned,Evaluator,load_data,channel_objective

STUDY=ROOT/'results/deep_learning/joint_private_readout_backbone_v141_20261001'
OUT=ROOT/'outputs/pure_dl_joint_private_readout_backbone_v141_fold5'
STAGES={'private_readouts':('TOTAL',512,False,0.,6e-4)}

class PrivateReadouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([nn.Sequential(nn.Linear(96,192),nn.GLU(-1),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,25)) for _ in range(3)])
        self.bias=nn.Parameter(torch.zeros(75))
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)

    def forward(self,h):
        outputs=[branch(h) for branch in self.branches]
        gates=torch.stack([y[...,0] for y in outputs],-1)
        density=torch.cat([y[...,1:] for y in outputs],-1)
        return torch.cat((gates,density),-1)+self.bias

class Network(Aligned):
    def __init__(self,master):
        super().__init__(master);self.output=PrivateReadouts()

    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(inference_prediction_capacity_same_as_v130=False,
            private_nonlinear_readouts=True,readout_description='three separate96->192 GLU96->LayerNorm/GELU->25 branches',
            same_initial_predictive_prior_as_aligned=True,
            new_student_no_trained_aligned_weights=True,architecture_and_capacity_change=True)
        return info

def make_model(stage,master,metadata):return Network(master)

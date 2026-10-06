"""Curve-family tokenization in the positive train-only-aligned new backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from joint_representation_backbone_v132r1 import ROOT,SOURCE,Network as Aligned,Evaluator,load_data,channel_objective
STUDY=ROOT/'results/deep_learning/joint_curve_backbone_v135_20261001'
OUT=ROOT/'outputs/pure_dl_joint_curve_backbone_v135_fold5'
STAGES={'curve_tokens':('TOTAL',512,False,0.,6e-4)}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
class CurveStem(nn.Module):
    def __init__(self,names):
        super().__init__();assert len(names)==206
        self.group_indices=[];self.encoders=nn.ModuleList()
        for i,curve in enumerate(CURVES):
            indices=[j for j,n in enumerate(names) if curve in n.replace('DENSITY','DEN').replace('SONIC','AC').split('_')]
            assert indices
            self.group_indices.append(indices);self.register_buffer('indices'+str(i),torch.tensor(indices,dtype=torch.long))
            self.encoders.append(nn.Sequential(nn.Linear(len(indices),32),nn.GELU(),nn.Linear(32,32)))
        self.identity=nn.Parameter(torch.randn(9,32)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(32,4,96,dropout=.1,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.combine=nn.Linear(288,96);self.shortcut=nn.Linear(206,96);self.relative=nn.Linear(1,96)
    def forward(self,inp):
        b,l,_=inp.shape;x=inp[...,:206]
        tokens=torch.stack([enc(x.index_select(-1,getattr(self,'indices'+str(i)))) for i,enc in enumerate(self.encoders)],2)
        z=(tokens+self.identity[None,None]).reshape(b*l,9,32)
        for block in self.blocks:z=block(z)
        return self.combine(z.reshape(b,l,288))+self.shortcut(x)+self.relative(inp[...,206:207])
class Network(Aligned):
    def __init__(self,master):
        super().__init__(master);self.stem=CurveStem(master['selected_feature_names'])
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(inference_prediction_capacity_same_as_v130=False,curve_family_tokenization=True,
            curve_order=list(CURVES),group_feature_indices=self.stem.group_indices,
            all206features_retained_in_dense_shortcut=True,no_new_input_features=True,
            perpoint_crosscurve_attention_layers=2,percurve_width=32,representation_change_plus_capacity=True)
        return info
def make_model(stage,master,metadata):return Network(master)

"""Frozen private SW backbone; matched likelihood versus joint-action utility head fitting."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_dropout_extend_v77 import ROOT,Network as Parent,Evaluator,load_data as parent_data
from train_dual8222_heads80 import SOURCE as PREPROCESS_SOURCE
from train_scaling_v21 import channel_objective
from por_t3_patch_loss import probability

STUDY=ROOT/'results/deep_learning/sw_head_utility_v80_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_input_products_v79_20260930'
SOURCE=ROOT/'outputs/pure_dl_sw_dropout_extend_v77_SW_p40_fold5/model_best.pt'
MODES={mode:dict(target='SW',c=2,length=1024,rate=0.,utility_weight=alpha) for mode,alpha in (('likelihood',0.),('joint',.5))}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def load_data():
    data,meta=parent_data();master=torch.load(PREPROCESS_SOURCE,map_location='cpu',weights_only=False)
    meta.update(sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]),
        preprocessing_checkpoint=str(PREPROCESS_SOURCE.relative_to(ROOT)),preprocessing_sha256=hashlib.sha256(PREPROCESS_SOURCE.read_bytes()).hexdigest())
    return data,meta

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('p40',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.requires_grad_(False);self.output.requires_grad_(True);self.study_mode=mode
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],initialization='complete v77 p40 SW function, no output reset',
            optimizer_reset=True,only_output_linear_trainable=True,trainable_parameters=3225,added_parameters=0,
            frozen_backbone_eval_mode=True,training_key_drop_probability=0.,evaluation_key_drop_probability=0.)
        self.train(False)
    def initialize_priors(self,data):return self.parent_info
    def train(self,mode=True):
        super().train(False);self.training=mode;self.output.train(mode);return self

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();self.alpha=MODES[mode]['utility_weight'];self.temperature=.05
        self.register_buffer('actions',torch.as_tensor(evaluator.actions,dtype=torch.float32))
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.target_median=metadata['sw_target_median'];self.target_scale=metadata['sw_target_scale']
    def forward(self,output,labels,default,mask,quality):
        likelihood,_,_=channel_objective(output,labels,default,mask,quality,2)
        gate,raw=output;g=gate[mask][:,0].float().sigmoid();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        conditional=probability(2,r[:,0].softmax(-1),r[:,1],torch.nn.functional.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g[:,None],(1-g[:,None])*conditional),-1)
        predicted_utility=joint@self.utilities
        action_probability=(predicted_utility/self.temperature).softmax(-1)
        physical_y=torch.exp(z*self.target_scale+self.target_median)
        physical_y=torch.where(d,torch.full_like(physical_y,99.9),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        reward=(action_probability*true_utility).sum(-1)
        decision=(q*(1-reward)).mean()/3
        # Both arms calculate and traverse this graph; alpha alone changes the loss.
        return likelihood+self.alpha*decision,likelihood,decision

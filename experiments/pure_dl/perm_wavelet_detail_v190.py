"""Masked private PERM detail: two-level Haar versus exactly matched long29."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from perm_private_width128_v185 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_wavelet_detail_v190_20261001'
OUT=ROOT/'outputs/pure_dl_perm_wavelet_detail_v190_fold5'
STAGES={k:('PERM',256,True,0.,6e-4) for k in ('wavelet','long29')}
class HaarDetail(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.width=base.in_channels
        self.filters=nn.ModuleList([nn.Conv1d(2*self.width,2*self.width,5,padding=2,groups=2*self.width,bias=False) for _ in range(2)])
        self.gains=nn.ParameterList([nn.Parameter(torch.full((1,self.width,2,1),.1)) for _ in range(2)])
        for layer in self.filters:nn.init.zeros_(layer.weight)
    def forward(self,x):
        current=x;parts=[]
        for filt,gain in zip(self.filters,self.gains):
            length=current.shape[-1]
            if length%2:current=F.pad(current,(0,1))
            even,odd=current[...,0::2],current[...,1::2]
            low=(even+odd)/math.sqrt(2);high=(even-odd)/math.sqrt(2)
            bands=torch.stack((low,high),2)
            processed=filt(bands.flatten(1,2)).reshape_as(bands)*gain
            parts.append((processed[:,:,0],processed[:,:,1],length))
            current=low
        reconstructed=None
        for low,high,length in reversed(parts):
            if reconstructed is not None:low=low+reconstructed
            even=(low+high)/math.sqrt(2);odd=(low-high)/math.sqrt(2)
            reconstructed=torch.stack((even,odd),-1).flatten(-2)[...,:length]
        return self.base(x)+reconstructed
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.study_mode=mode
        base=self.body.detail[0].depthwise
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            if mode=='wavelet':new=HaarDetail(base)
            elif mode=='long29':
                new=nn.Conv1d(128,128,29,padding=14,groups=128,bias=True)
                with torch.no_grad():
                    new.weight.zero_();new.weight[:,:,12:17].copy_(base.weight);new.bias.copy_(base.bias)
            else:raise ValueError(mode)
        self.body.detail[0].depthwise=new
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(detail_variant=self.study_mode,only_first_detail_depthwise_changed=True,
                    extra_active_parameters=3072,matched_extra_capacity=True,control_width=128,
                    random_hidden_initialization_differs=False,
                    base128_weights_same_initialization=True,extra_filters_zero_initialized=True,
                    same_initial_hidden_and_predictive_function=True,Haar_levels=2 if self.study_mode=='wavelet' else 0,
                    long_kernel=29 if self.study_mode=='long29' else 5,
                    no_trained_v185_weights_loaded=True,phase_alignment='fixed window row pairing',
                    valid_mask_before_and_after_depthwise=True,unused_padded_inputs_zero=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)

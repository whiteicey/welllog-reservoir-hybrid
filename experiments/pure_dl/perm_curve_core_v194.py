"""Local nine-curve STAR-style core: mean versus weighted aggregation."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_private_width128_v185 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_curve_core_v194_20261001'
OUT=ROOT/'outputs/pure_dl_perm_curve_core_v194_fold5'
STAGES={k:('PERM',256,True,0.,6e-4) for k in ('mean','weighted')}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')

class CurveCore(nn.Module):
    def __init__(self,indices,mode):
        super().__init__();self.mode=mode
        self.register_buffer('indices',torch.tensor(indices,dtype=torch.long))
        self.embed=nn.Linear(16,64);self.curve_id=nn.Parameter(torch.zeros(9,64))
        self.gen1=nn.Linear(64,64);self.gen2=nn.Linear(64,32)
        self.gen3=nn.Linear(96,64);self.gen4=nn.Linear(64,64)
        self.project=nn.Linear(9*64,128)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,x,valid):
        b,l,_=x.shape;assert l%16==0
        values=x.index_select(-1,self.indices).masked_fill(~valid[...,None],0)
        patches=values.reshape(b,l//16,16,9).transpose(-1,-2)
        tokens=F.gelu(self.embed(patches))+self.curve_id
        candidate=self.gen2(F.gelu(self.gen1(tokens)))
        core=(candidate.softmax(-2)*candidate).sum(-2,keepdim=True) if self.mode=='weighted' else candidate.mean(-2,keepdim=True)
        distributed=self.gen4(F.gelu(self.gen3(torch.cat((tokens,core.expand(-1,-1,9,-1)),-1))))
        residual=.1*self.project(distributed.flatten(-2))
        residual=residual.masked_fill(~valid.reshape(b,l//16,16).any(-1)[...,None],0)
        return F.interpolate(residual.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2).masked_fill(~valid[...,None],0)

class Body(nn.Module):
    def __init__(self,original,indices,mode):
        super().__init__()
        for name in ('stem','local','token_pos','global_blocks','fuse','detail'):setattr(self,name,getattr(original,name))
        self.curve_core=CurveCore(indices,mode)
    def forward(self,x,relative,valid,generator=None):
        b,l,_=x.shape
        h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=(h+self.curve_core(x,valid)).masked_fill(~valid[...,None],0)
        for block in self.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,128)*pv[...,None]).sum(2)/n[...,None]
        positions=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,positions),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+self.token_pos(loc)).masked_fill(~tv[...,None],0)
        for block in self.global_blocks:token=block(token,tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(self.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        for block in self.detail:h=block(h,valid)
        return h,h.new_zeros(()),None

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.study_mode=mode
        names=list(master['selected_feature_names']);indices=[names.index(name) for name in CURVES]
        assert len(names)==206 and len(set(indices))==9
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.body=Body(self.body,indices,mode)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(local_curve_core=self.study_mode,curve_names=list(CURVES),curve_indices=self.body.curve_core.indices.tolist(),
                    patch_length=16,series_width=64,core_width=32,only_input_side_branch=True,
                    curve_communication_then_redistribution=True,stochastic_pooling=False,
                    full_resolution_main_path_preserved=True,zero_project=True,residual_scale=.1,
                    same_initial_hidden_and_predictive_function=True,same_initial_parameter_tensors=True,
                    no_trained_v185_weights_loaded=True,not_exact_SOFTS_reproduction=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)

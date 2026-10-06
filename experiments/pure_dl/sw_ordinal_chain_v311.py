"""Conditional probability-chain parameterization, original grid retained."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_ordinal_chain_v311_20261002'
OUT=ROOT/'outputs/pure_dl_sw_ordinal_chain_v311_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('ordered','shuffled')}

class ConditionalReadout(nn.Module):
    def __init__(self,linear,mode):
        super().__init__();self.linear=linear
        # All256 output rows participate; common shift is the redundant direction.
        nn.init.zeros_(linear.weight);nn.init.zeros_(linear.bias)
        with torch.no_grad():linear.bias[:255].copy_(torch.arange(255,0,-1,dtype=linear.bias.dtype).log())
        permutation=torch.arange(256) if mode=='ordered' else torch.randperm(256,generator=torch.Generator().manual_seed(20261311))
        self.register_buffer('physical_inverse',permutation.argsort())
    def forward(self,h):
        # Preserve uniform-bias initialization and long-chain arithmetic in FP32.
        with torch.autocast(device_type=h.device.type,enabled=False):
            ell=F.linear(h.float(),self.linear.weight.float(),self.linear.bias.float())
        a=ell[...,:255]-ell[...,255:]
        log_survive=F.logsigmoid(a);cum=log_survive.cumsum(-1)
        exclusive=torch.cat((torch.zeros_like(cum[...,:1]),cum[...,:-1]),-1)
        logmass=torch.cat((exclusive+F.logsigmoid(-a),cum[...,-1:]),-1)
        logmass=logmass.index_select(-1,self.physical_inverse)
        return logmass-logmass.logsumexp(-1,keepdim=True)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.chain_mode=mode
        self.readout=ConditionalReadout(self.readout,mode)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        active=sum(p.numel() for p in self.parameters() if p.requires_grad);assert active==65265
        info.update(chain_mode=self.chain_mode,new_active_parameters=active,conditional_thresholds=255,
                    all256_linear_rows_active=True,nonzero_readout_initialization=False,uniform_initial_new_mass=True,
                    new_readout_zero_weight_uniform_bias=True,physical_grid_and_decoder_unchanged=True,
                    only_original104_weights_loaded=True,no297_or310_weights_loaded=True,
                    reference_row_gradient_coupling=True,not_full_CORN_loss_reproduction=True,
                    ordinal_BCE_added=False,one_SW_additional_phase=True)
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

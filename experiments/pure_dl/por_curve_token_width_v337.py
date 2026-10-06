"""Fresh joint POR raw encoder/readout with width128 nine-curve interaction."""
import torch
from torch import nn
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent,CurveInteraction
STUDY=ROOT/'results/deep_learning/por_curve_token_width_v337_20261002'
OUT=ROOT/'outputs/pure_dl_por_curve_token_width_v337_fold5'
STAGES={'wide':('POR',256,False,0.,3e-4)}
class WideBlock(nn.Module):
 def __init__(self):
  super().__init__();self.n1=nn.LayerNorm(128);self.attention=nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
  self.n2=nn.LayerNorm(128);self.ff=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
 def forward(self,h,mask):
  z=self.n1(h);h=h+self.attention(z,z,z,attn_mask=mask,need_weights=False)[0]
  return h+self.ff(self.n2(h))
class WideCurveInteraction(CurveInteraction):
 def __init__(self):
  nn.Module.__init__(self);self.mode='full';self.numeric=nn.Linear(1,128);self.identity=nn.Parameter(torch.randn(9,128)*.02)
  self.blocks=nn.ModuleList([WideBlock(),WideBlock()]);self.bridge=nn.Linear(128,48)
  nn.init.zeros_(self.bridge.weight);nn.init.zeros_(self.bridge.bias)
  groups=torch.tensor([0,0,0,2,1,1,1,2,2]);self.register_buffer('group_mask',groups[:,None]!=groups[None,:])
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'full')
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261302);self.curve_interaction=WideCurveInteraction()
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);attention=sum(p.numel() for p in self.curve_interaction.parameters());assert n==330128 and attention==272560
  info.update(token_width=128,attention_blocks=2,heads=4,FFN=256,new_interaction_parameters=attention,new_active_parameters=n,active_parameters=n,
   raw_encoder_readout_width_unchanged=48,bridge_dimensions=[128,48],initial_predictor_preserved_by_zero_bridge_and_alpha0=True,
   original151_only=True,no335_or_other_trained_repair_weights=True,all_fresh_raw_encoder_attention_readout_joint=True,
   historical_reference='335wide64 active128336/attention70768 best28 natural42',capacity_matched=False,one_POR_added_phase=True,
   seed_for_new_attention=20261302,changed_weight_shapes_and_initial_hidden_functions=True,limited_history_no_exact128_intersection_found=True,only_scope_differs_in_matched_pair=False)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

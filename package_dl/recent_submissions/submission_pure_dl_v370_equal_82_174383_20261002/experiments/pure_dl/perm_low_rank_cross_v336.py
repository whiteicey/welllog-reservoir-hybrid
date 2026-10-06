"""Joint PERM encoder/readout with low-rank cross versus additive feature adapter."""
import torch
from torch import nn
from perm_group_shape_v312 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from perm_sparse_reacquisition_v291 import Network as SparseParent
STUDY=ROOT/'results/deep_learning/perm_low_rank_cross_v336_20261002'
OUT=ROOT/'outputs/pure_dl_perm_low_rank_cross_v336_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('cross','additive')}
class FeatureAdapter(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48,elementwise_affine=False);self.down=nn.Linear(48,8,bias=False);self.up=nn.Linear(8,48)
  nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,h,valid):
  with torch.autocast(device_type=h.device.type,enabled=False):
   z=self.norm(h.float());u=self.up(self.down(z));term=z*u if self.mode=='cross' else u
   return ((h.float()+.25*torch.tanh(term))*valid.unsqueeze(-1)).to(h.dtype)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'rational');self.cross_mode=mode;self.current_valid=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261336);self.feature_adapter=FeatureAdapter(mode)
  self.stem.register_forward_hook(self.adapt_stem)
 def adapt_stem(self,module,args,h):return self.feature_adapter(h,self.current_valid)
 def groups(self):return super().groups()+[dict(params=list(self.feature_adapter.parameters()),lr=3e-4,name='PERM_joint_low_rank_feature_adapter')]
 def initialize_priors(self,data):
  info=SparseParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==37386
  info.update(shape_mode='rational',new_shape_parameters=104,new_low_rank_adapter_parameters=816,new_active_parameters=n,active_parameters=n,
   cross_mode=self.cross_mode,rank=8,zero_initial_up_projection=True,down_bias=False,affine_free_LN=True,residual_bound=.25,
   feature_adapter_before_first_selector=True,original154_only=True,no291_or312_or306_trained_weights=True,
   one_PERM_additional_phase=True,joint_new_body_shape_adapter_readout=True,
   previous_POR306_pair_positive_not_absolute_best=True,not_first_bilinear_or_DCN_class=True,not_full_DCNv2_reproduction=True,
   primary_control='same816parameters/init/upzero, z-times-lowrank(z) versus additive-lowrank(z); effective functionfamilies differ')
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_valid=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)

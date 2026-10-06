"""Joint PERM encoder/readout with fixed valid-depth nonlinear edge messages."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_group_shape_v312 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from perm_sparse_reacquisition_v291 import Network as SparseParent
STUDY=ROOT/'results/deep_learning/perm_edge_messages_v333_20261002'
OUT=ROOT/'outputs/pure_dl_perm_edge_messages_v333_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('maximum','mean')}
class EdgeMessages(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48,elementwise_affine=False);self.message=nn.Linear(96,24);self.output=nn.Linear(24,48)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,h,valid):
  with torch.autocast(device_type=h.device.type,enabled=False):
   z=self.norm(h.float());rows=torch.arange(h.shape[1],device=h.device);aggregate=None;count=torch.zeros_like(valid,dtype=torch.float32)
   for offset in range(-3,4):
    ix=rows+offset;in_range=(ix>=0)&(ix<h.shape[1]);ix=ix.clamp(0,h.shape[1]-1);neighbor=z.index_select(1,ix);mask=valid&valid.index_select(1,ix)&in_range[None]
    message=F.gelu(self.message(torch.cat((z,neighbor-z),-1)))
    if self.mode=='maximum':
     message=message.masked_fill(~mask.unsqueeze(-1),-float('inf'));aggregate=message if aggregate is None else torch.maximum(aggregate,message)
    else:
     message=message*mask.unsqueeze(-1);aggregate=message if aggregate is None else aggregate+message;count=count+mask.float()
   if self.mode=='mean':aggregate=aggregate/count.clamp_min(1).unsqueeze(-1)
   aggregate=torch.where(valid.unsqueeze(-1),aggregate,torch.zeros_like(aggregate))
   return ((h.float()+.25*torch.tanh(self.output(aggregate)))*valid.unsqueeze(-1)).to(h.dtype)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'rational');self.edge_mode=mode;self.current_valid=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261333);self.edge_messages=EdgeMessages(mode)
  self.stem.register_forward_hook(self.add_edge_messages)
 def add_edge_messages(self,module,args,h):return self.edge_messages(h,self.current_valid)
 def groups(self):return super().groups()+[dict(params=list(self.edge_messages.parameters()),lr=3e-4,name='PERM_joint_depth_edge_messages')]
 def initialize_priors(self,data):
  info=SparseParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==40098
  info.update(shape_mode='rational',new_shape_parameters=104,new_edge_message_parameters=3528,new_active_parameters=n,active_parameters=n,
   edge_mode=self.edge_mode,edge_span_rows=7,edge_offsets=list(range(-3,4)),message_width=24,message_input='normalized center concatenated with neighbor-minus-center',
   initial_function_preserved_by_zero_output=True,affine_free_LN=True,edge_before_first_selector=True,valid_samewell_depth_neighbors_only=True,no_physical_gap_compensation=True,
   original154_only=True,no291_or312_or325_trained_weights=True,one_PERM_additional_phase=True,joint_body_shape_edge_readout=True,
   not_dynamic_KNN_or_retrieval=True,not_full_DGCNN_reproduction=True,
   primary_control='same parameters/initialization/localmessages, componentwise max versus masked mean aggregation; effective function families differ')
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_valid=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)

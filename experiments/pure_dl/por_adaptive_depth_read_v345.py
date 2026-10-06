"""One sparse content-driven depth reader in fresh positive POR coupling code."""
import torch
from torch import nn
from por_coupling_conditional_rps_v341 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_adaptive_depth_read_v345_20261002'
OUT=ROOT/'outputs/pure_dl_por_adaptive_depth_read_v345_fold5'
STAGES={s:('POR',256,False,0.,3e-4) for s in ('adaptive','fixed')}
class SparseDepthRead(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(64);self.select=nn.Linear(64,10);self.value=nn.Linear(64,8);self.up=nn.Linear(8,64)
  nn.init.zeros_(self.select.weight);nn.init.zeros_(self.select.bias);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,h,valid):
  b,l,n,_=h.shape;v=self.norm(h);control=self.select(v).float();z=self.value(v).float()*valid[:,:,None,None]
  offsets=2*torch.tanh(control[...,:5]) if self.mode=='adaptive' else torch.zeros_like(control[...,:5])
  logits=control[...,5:] if self.mode=='adaptive' else (control[...,:5]+control[...,5:])*.5
  base=torch.arange(l,device=h.device,dtype=torch.float32)[None,:,None];vm=valid[:,:,None].expand(-1,-1,n)
  samples=[];available=[]
  for tap,ref in enumerate((-8,-2,0,2,8)):
   position=base+ref+offsets[...,tap];a=position.floor().long();c=a+1;fraction=position-a.float()
   ai=a.clamp(0,l-1);ci=c.clamp(0,l-1)
   am=(a>=0)&(a<l)&torch.gather(vm,1,ai);cm=(c>=0)&(c<l)&torch.gather(vm,1,ci)
   wa=(1-fraction)*am;wc=fraction*cm;den=wa+wc
   za=torch.gather(z,1,ai[...,None].expand(-1,-1,-1,8));zc=torch.gather(z,1,ci[...,None].expand(-1,-1,-1,8))
   samples.append((wa[...,None]*za+wc[...,None]*zc)/den.clamp_min(1e-6)[...,None]);available.append(den>0)
  avail=torch.stack(available,-1);has=avail.any(-1)
  safe=logits.masked_fill(~avail,-float('inf'));safe=torch.where(has[...,None],safe,torch.zeros_like(safe))
  weights=safe.softmax(-1)*avail;result=(torch.stack(samples,-2)*weights[...,None]).sum(-2)
  # Short isolated valid queries can lose every shifted tap: use their own value.
  result=torch.where(has[...,None],result,z)
  return h+.25*torch.tanh(self.up(result.to(h.dtype)))*valid[:,:,None,None]
class ReaderInteraction(nn.Module):
 def __init__(self,old,mode):
  super().__init__();self.numeric=old.numeric;self.identity=old.identity;self.blocks=old.blocks;self.bridge=old.bridge;self.temporal=old.temporal
  self.register_buffer('group_mask',old.group_mask);self.current_valid=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261345);self.reader=SparseDepthRead(mode)
 def forward(self,curves):
  b,l=curves.shape[:2];h=self.numeric(curves.reshape(-1,9,1))+self.identity[None]
  h=self.blocks[0](h,None).reshape(b,l,9,64);assert self.current_valid is not None
  h=self.reader(h,self.current_valid);h=self.temporal(h,self.current_valid).reshape(-1,9,64);h=self.blocks[1](h,None)
  return self.bridge(h.mean(1)).reshape(b,l,48)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'conditional');self.read_mode=mode;self.curve_interaction=ReaderInteraction(self.curve_interaction,mode)
 def initialize_priors(self,data):
  # Parent 341 checks134752; temporarily omit fresh reader from prior bookkeeping.
  reader=self.curve_interaction.reader;del self.curve_interaction.reader
  try:info=super().initialize_priors(data)
  finally:self.curve_interaction.reader=reader
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in reader.parameters());assert n==136626 and extra==1874
  info.update(active_parameters=n,new_active_parameters=n,new_sparse_reader_parameters=extra,reader_mode=self.read_mode,references=[-8,-2,0,2,8],maximum_row_offset=2,not_physical_depth_invariant=True,low_rank_values=8,zero_reader_output=True,masked_endpoint_renormalization=True,all_unavailable_query_falls_back_to_own_value=True,only_original151_checkpoint=True,no341_orotherrepair_weights=True,all_fresh_encoder_reader_readout_joint=True,one_POR_added_phase=True,primary_control='adaptive vs fixed-reference contentweighted, same1874physicalparams/init but offset/weightroles andeffectivecapacity differ')
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

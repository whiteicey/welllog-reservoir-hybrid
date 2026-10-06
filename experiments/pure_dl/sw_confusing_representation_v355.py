"""Finite confusing-continuous representation supervision, no teacher or extra phase."""
import torch
from torch.nn import functional as F
from sw_dynamic_cross_spectrum_v351 import ROOT,SOURCE,PARENT,load_data,Evaluator,DecisionObjective,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_confusing_representation_v355_20261002'
OUT=ROOT/'outputs/pure_dl_sw_confusing_representation_v355_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('joint','detached')}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'aligned');self.conr_mode=mode;self.conr_cache=None
  self.conr_stats=dict(samples=0,eligible=0,positive_pairs=0,confusing_pairs=0,aux=0.)
  self.readout.register_forward_hook(self.capture_conr)
 def capture_conr(self,module,args,value):
  if self.training and self.conr_cache is None:self.conr_cache=(args[0],value)
 def forward(self,*args,**kwargs):
  self.conr_cache=None
  return super().forward(*args,**kwargs)
 def configure_conr(self,evaluator):
  self.register_buffer('conr_utilities',evaluator.utilities[1:].detach().clone())
  self.register_buffer('conr_actions',torch.as_tensor(evaluator.actions,device=self.mix_alpha.device,dtype=torch.float32))
  assert self.conr_utilities.shape[0]==256
 def conr(self,ix,mask,data):
  h,logits=self.conr_cache
  select=(mask & ~data['default'][ix,2] & (ix.remainder(32)==0)).flatten().nonzero().flatten()
  if select.numel():
   rows=ix.flatten()[select];order=rows.argsort(stable=True);select=select[order];rows=rows[order]
   unique=torch.cat((torch.ones(1,device=rows.device,dtype=torch.bool),rows[1:]!=rows[:-1]));select=select[unique];rows=rows[unique]
   order=((rows*1103515245+12345).remainder(2147483647)).argsort(stable=True);select=select[order];rows=rows[order]
   wells=data['row_well'][rows];earlier=torch.arange(len(rows),device=rows.device)
   rank=((wells[:,None]==wells[None,:])&(earlier[None,:]<=earlier[:,None])).sum(1)
   keep=rank<=8;select=select[keep][:256];rows=rows[keep][:256]
  z=h.reshape(-1,48)[select].float();self.conr_stats=dict(samples=len(select),eligible=0,positive_pairs=0,confusing_pairs=0,aux=0.)
  if self.conr_mode=='detached':z=z.detach()
  if len(select)<3:return z.sum()*0
  y=data['conr_truth'][rows];assert bool(torch.isfinite(y).all())
  with torch.no_grad():
   q=logits.reshape(-1,256)[select].float().softmax(-1);actions=self.conr_actions[(q@self.conr_utilities).argmax(-1)]
   off=~torch.eye(len(select),device=z.device,dtype=torch.bool)
   pos=((y[:,None]-y[None,:]).abs()<=.05*(y.abs()[:,None]+1e-5))&off
   neg=(~pos)&off&((actions[:,None]-actions[None,:]).abs()<=.05*(actions.abs()[:,None]+1e-5))
   eligible=pos.any(1)&neg.any(1)
  self.conr_stats.update(eligible=int(eligible.sum()),positive_pairs=int(pos.sum()),confusing_pairs=int(neg.sum()))
  if not bool(eligible.any()):return z.sum()*0
  zn=F.normalize(z,dim=-1);sim=(zn@zn.T)/.1
  selected=sim[eligible];p=pos[eligible];allowed=(pos|neg)[eligible]
  li=torch.logsumexp(selected.masked_fill(~allowed,-float('inf')),dim=1)-(selected*p).sum(1)/p.sum(1)
  quality=data['quality'][rows][eligible].detach().float();loss=(li*quality).sum()/quality.sum().clamp_min(1e-12)
  self.conr_stats['aux']=float(loss.detach());return loss
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(conr_mode=self.conr_mode,conr_added_parameters=0,conr_feature='actual new fuse48 density input',conr_temperature=.1,conr_coefficient=.1/3,conr_sampling='row%32 dedup hash/max8perwell/max256',conr_physical_tolerance=.05,conr_own_component_actions_only=True,no_teacher_pseudo_KL_or_validation_labels=True)
  return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)

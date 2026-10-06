"""Fresh compact PERM encoder/readout with content-selective bidirectional memory."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from perm_group_shape_v312 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as ShapeParent
from perm_sparse_reacquisition_v291 import Network as SparseParent,sparsemax
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/perm_selective_memory_v325_20261002'
OUT=ROOT/'outputs/pure_dl_perm_selective_memory_v325_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('selective','window')}
def affine_scan(a,b):
 # Inclusive affine prefixes. Right compose left, using old arrays at each stage.
 step=1
 while step<a.shape[1]:
  old_a,old_b=a,b
  a=torch.cat((old_a[:,:step],old_a[:,step:]*old_a[:,:-step]),dim=1)
  b=torch.cat((old_b[:,:step],old_b[:,step:]+old_a[:,step:]*old_b[:,:-step]),dim=1)
  step*=2
 return b
class SelectiveMemory(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48,elementwise_affine=False)
  self.value=nn.Linear(48,48);self.update=nn.Linear(48,48);self.output=nn.Linear(48,48)
  nn.init.zeros_(self.update.weight);nn.init.constant_(self.update.bias,math.log(.2/.8))
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,h,valid):
  with torch.autocast(device_type=h.device.type,enabled=False):
   z=self.norm(h.float());vf=valid.unsqueeze(-1).float()
   if self.mode=='window':z=((z*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)).expand_as(z)
   g=1/64+(1/4-1/64)*torch.sigmoid(self.update(z))
   u=F.silu(self.value(h.float()));a=(1-g)*vf;b=g*u*vf
   forward=affine_scan(a,b);reverse=affine_scan(a.flip(1),b.flip(1)).flip(1)
   correction=.25*torch.tanh(self.output((forward+reverse)*.5)/.25)*vf
   return (h.float()+correction).to(h.dtype)
class Network(ShapeParent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'rational');self.memory_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261325);self.depth_memory=SelectiveMemory(mode)
 def groups(self):return super().groups()+[dict(params=list(self.depth_memory.parameters()),lr=3e-4,name='PERM_joint_selective_depth_memory')]
 def initialize_priors(self,data):
  info=SparseParent.initialize_priors(self,data)
  count=sum(p.numel() for p in self.parameters() if p.requires_grad);assert count==43626
  info.update(shape_mode='rational',new_active_parameters=count,new_shape_parameters=104,new_memory_parameters=7056,
   memory_mode=self.memory_mode,memory_after_first_GLU=True,bidirectional_shared_parameters=True,
   gate_bounds=[1/64,1/4],initial_gate=1/16,initial_memory_residual_zero=True,scan_stages=8,
   associative_scan_work='O(LlogL), FP32 states, no cumprod division, invalid resets',
   original154_only=True,no291_or312_trained_weights_loaded=True,
   one_PERM_additional_phase=True,joint_body_shape_memory_readout=True,
   primary_control='same memory/parameters/init, point-content versus valid-window-mean gate input')
  return info
 def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
     self.aux_logits=None
     with torch.no_grad():
         gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
         flat=raw[:,:,0].float().reshape(-1,3,16)
         prior=torch.cat([probability(1,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
     inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
     h=self.stem(inp).masked_fill(~valid.unsqueeze(-1),0)
     curves=x.index_select(-1,self.curve_indices);budget=torch.ones_like(curves,dtype=torch.float32)
     vf=valid.unsqueeze(-1).float();supports=[];variations=[]
     for block,(selector,transform) in enumerate(zip(self.selectors,self.transforms)):
         logits=selector(h).float()
         if self.mode=='window':logits=((logits*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)).expand_as(logits)
         mask=sparsemax(budget*logits);budget=budget*(1.3-mask)
         selected=(9*mask*curves.float()).to(h.dtype)
         h=((h+F.glu(transform(torch.cat((h,selected),-1)),-1))/math.sqrt(2)).masked_fill(~valid.unsqueeze(-1),0)
         if block==0:h=self.depth_memory(h,valid)
         if self.training:
             supports.append(((mask>0).float()*vf).sum()/vf.sum().clamp_min(1))
             mean=(mask*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)
             variations.append(((mask-mean).square()*vf).sum()/vf.sum().clamp_min(1))
     logits=self.readout(h).float()
     if self.training:
         self.aux_logits=[logits]
         self.last_support=float(torch.stack(supports).mean().detach())
         self.last_mask_variation=float(torch.stack(variations).mean().detach())
     mass=((1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)).clamp_min(1e-12)
     mass=mass/mass.sum(-1,keepdim=True)
     return gate,mass.log().unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,metadata,stage)

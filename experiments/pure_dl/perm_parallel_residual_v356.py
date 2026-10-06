"""Preserve native154 predictor; learn a parallel local backbone and distribution residual."""
import hashlib,torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,load_data,Evaluator,channel_objective as native_loss,Network as Native
STUDY=ROOT/'results/deep_learning/perm_parallel_residual_v356_20261002'
OUT=ROOT/'outputs/pure_dl_perm_parallel_residual_v356_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('local','point')}
class ResidualEncoder(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.stem=nn.Conv1d(319,48,1);self.norm=nn.LayerNorm(48);self.depth=nn.Conv1d(48,48,17,padding=8,groups=48);self.expand=nn.Conv1d(48,96,1)
 def forward(self,x,valid):
  v=valid[:,None];h=F.gelu(self.stem(x)).masked_fill(~v,0);z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~v,0)
  if self.mode=='local':z=self.depth(z)
  else:z=F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=48)
  a,b=self.expand(F.gelu(z).masked_fill(~v,0)).chunk(2,1)
  return (h+F.gelu(a)*b.sigmoid()).masked_fill(~v,0)
class Network(nn.Module):
 def __init__(self,master,metadata,mode):
  super().__init__();self.mode=mode;self.base=Native(master,'perm_conditional')
  assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
  self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.base.requires_grad_(False).eval()
  self.register_buffer('rps_edges',self.base.rps_edges.detach().clone());self.register_buffer('default_z',self.base.default_z.detach().clone())
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261356);self.residual_encoder=ResidualEncoder(mode);self.residual_readout=nn.Conv1d(48,32,1)
   nn.init.zeros_(self.residual_readout.weight);nn.init.zeros_(self.residual_readout.bias)
  self.residual_input=None;self.base.context_head.register_forward_pre_hook(self.capture_residual_input)
  self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.
 def capture_residual_input(self,module,args):self.residual_input=args[0].detach()
 def train(self,mode=True):super().train(mode);self.base.eval();return self
 def groups(self):return [dict(params=list(self.residual_encoder.parameters()),lr=3e-4,name='PERM_parallel_residual_feature_encoder'),dict(params=list(self.residual_readout.parameters()),lr=3e-4,name='PERM_new_distribution_residual_readout')]
 def initialize_priors(self,data):
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==22592
  return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,original154_function_fully_frozen=True,zero_new_readout=True,initial_function_exact154=True,
   input='original154 two128 expert features plus original63 side features',new_encoder_width=48,stored_depth_kernel=17,effective_depth_kernel=17 if self.mode=='local' else 1,
   new_encoder_readout_joint=True,new_active_parameters=n,location_residual_bound_sd=.25,logweight_residual_bound=.25,gate_scale_unchanged=True,
   own_Gaussian16_NLL_BCE_RPS1_decoder=True,no_trained_repair_weights=True,one_added_PERM_phase=True,primary_control='same22592params/init local17 versus point sum of same stored kernel; effectivefunction capacity differs',
   historical349rank4enc2negative=True,historicalnative154_contextpoint319to128retained=True)
 def forward(self,x,relative,valid,**kwargs):
  self.residual_input=None
  with torch.no_grad():gate,raw=self.base(x,relative,valid,**kwargs)
  assert self.residual_input is not None
  h=self.residual_encoder(self.residual_input,valid);self.residual_input=None
  correction=self.residual_readout(h).transpose(1,2).float();own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
  delta=.25*reference*torch.tanh(correction[...,:16]);weights=.25*torch.tanh(correction[...,16:])
  corrected=torch.stack((own[:,:,0]+weights,own[:,:,1]+delta,own[:,:,2]),2).unsqueeze(2)
  return gate,corrected
def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):return native_loss(output,labels,default,mask,quality,c,'perm_conditional',model)

"""Fresh positive PERM backbone/readout with ordered numerical PLE residual."""
import numpy as np
import torch
from torch import nn
from perm_low_rank_cross_v336 import ROOT,SOURCE,PARENT,PARENT_SHA,Evaluator,channel_objective,Network as Parent,load_data as original_data
from perm_sparse_reacquisition_v291 import Network as SparseParent
STUDY=ROOT/'results/deep_learning/perm_ordered_knots_v346_20261002'
OUT=ROOT/'outputs/pure_dl_perm_ordered_knots_v346_fold5'
STAGES={s:('PERM',256,False,0.,3e-4) for s in ('learned','fixed')}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
def load_data(stage):
 data,master,metadata=original_data(stage);indices=[list(master['selected_feature_names']).index(n) for n in CURVES]
 values=data['x'].index_select(-1,torch.tensor(indices,device=data['x'].device))[data['train_raw']].detach().cpu().numpy();bounds=[];active=[];info=[]
 for i,v in enumerate(values.T):
  v=v[np.isfinite(v)];span=float(np.ptp(v)) if len(v) else 0.;tau=max(1e-6,span*1e-6)
  lo,hi=(np.quantile(v,[.05,.95]).tolist() if len(v) else [0.,0.]);fallback=False
  if hi-lo<=tau and span>tau:lo,hi=float(v.min()),float(v.max());fallback=True
  enabled=hi-lo>tau
  info.append(dict(curve=CURVES[i],low=float(lo),high=float(hi),full_span=span,tolerance=tau,fullrange_fallback=fallback,active=enabled))
  if enabled:active.append(indices[i]);bounds.append([float(lo),float(hi)])
 assert bounds,'No variable original input curve'
 metadata.update(numeric_bounds=bounds,numeric_active_indices=active,numeric_fit_info=info,numeric_fit_rows=len(values),numeric_fit_population='original train_raw anchors in64training wells, inputs only')
 return data,master,metadata
class OrderedPLE(nn.Module):
 def __init__(self,metadata,mode):
  super().__init__();self.mode=mode;self.k=8;self.register_buffer('indices',torch.tensor(metadata['numeric_active_indices'],dtype=torch.long));self.register_buffer('bounds',torch.tensor(metadata['numeric_bounds'],dtype=torch.float32));self.width_logits=nn.Parameter(torch.zeros(len(self.indices),8),requires_grad=mode=='learned');self.projection=nn.Linear(len(self.indices)*8,48,bias=False);nn.init.zeros_(self.projection.weight)
 def widths(self):
  eps=1/(64*self.k);span=self.bounds[:,1]-self.bounds[:,0]
  return span[:,None]*(eps+(1-self.k*eps)*self.width_logits.float().softmax(-1))
 def forward(self,x):
  with torch.autocast(device_type=x.device.type,enabled=False):
   widths=self.widths();left=self.bounds[:,0,None]+torch.cat((torch.zeros_like(widths[:,:1]),widths.cumsum(-1)[:,:-1]),-1)
   values=x.index_select(-1,self.indices).float().unsqueeze(-1);phi=((values-left)/widths).clamp(0,1).flatten(-2)
   return .25*torch.tanh(self.projection(phi))
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'cross');self.numeric_metadata=metadata;self.numeric_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261346);self.numeric=OrderedPLE(metadata,mode)
  self.stem[0].register_forward_hook(self.add_numeric)
 def add_numeric(self,module,args,h):return h+self.numeric(args[0]).to(h.dtype)
 def groups(self):return super().groups()+[dict(params=[p for p in self.numeric.parameters() if p.requires_grad],lr=3e-4,name='PERM_joint_ordered_numeric_basis')]
 def initialize_priors(self,data):
  info=SparseParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);s=len(self.numeric.indices)*8;expected=37386+48*s+(s if self.numeric_mode=='learned' else 0);assert n==expected
  info.update(active_parameters=n,new_active_parameters=n,positive_shape_mode='rational',new_shape_parameters=104,new_low_rank_cross_parameters=816,new_numeric_projection_parameters=48*s,new_numeric_trainable_width_parameters=s if self.numeric_mode=='learned' else 0,numeric_mode=self.numeric_mode,numeric_fit_info=self.numeric_metadata['numeric_fit_info'],numeric_fit_rows=self.numeric_metadata['numeric_fit_rows'],numeric_fit_population=self.numeric_metadata['numeric_fit_population'],numeric_active_indices=self.numeric.indices.tolist(),numeric_active_dimensions=s,numeric_train_only_input_fit=True,numeric_initial_uniform_segments=8,numeric_knots_ordered_endpoints_fixed=True,minimum_interval_fraction=1/512,zero_numeric_projection=True,numeric_stem_linear_before_GELU=True,original207scalar_missing_relative_bypass_retained=True,original154_only=True,no336_or276_orotherrepair_weights=True,all_fresh_encoder_shape_cross_numeric_readout_joint=True,one_PERM_added_phase=True,primary_control='sameuniforminitialknots/projection; learned addsS trainablewidthlogits, nottrainablecapacitymatched; no dummy parameters')
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

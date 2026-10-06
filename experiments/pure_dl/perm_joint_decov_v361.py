"""Finite covariance regularization of the positive fresh PERM encoder."""
import torch
from perm_low_rank_cross_v336 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent

STUDY=ROOT/'results/deep_learning/perm_joint_decov_v361_20261002'
OUT=ROOT/'outputs/pure_dl_perm_joint_decov_v361_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('decorrelation','identity')}

class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'cross')
  self.decov_mode=mode;self.decov_features=None
  self.decov_stats=dict(samples=0,trace=0.,offdiag_ratio=0.,main_loss=0.,auxiliary_loss=0.)
  self.readout.register_forward_pre_hook(self.capture_decov_input)
 def capture_decov_input(self,module,args):
  if self.training:self.decov_features=args[0]
 def forward(self,*args,**kwargs):
  self.decov_features=None
  return super().forward(*args,**kwargs)
 def decorrelation(self,ix,mask,data):
  assert self.decov_features is not None
  h=self.decov_features
  select=(mask & ~data['default'][ix,1] & (ix.remainder(32)==0)).flatten().nonzero().flatten()
  if select.numel():
   rows=ix.flatten()[select];order=rows.argsort(stable=True);select=select[order];rows=rows[order]
   unique=torch.cat((torch.ones(1,device=rows.device,dtype=torch.bool),rows[1:]!=rows[:-1]))
   select=select[unique];rows=rows[unique]
   order=((rows*1103515245+12345).remainder(2147483647)).argsort(stable=True)
   select=select[order];rows=rows[order]
   wells=data['row_well'][rows];earlier=torch.arange(len(rows),device=rows.device)
   rank=((wells[:,None]==wells[None,:])&(earlier[None,:]<=earlier[:,None])).sum(1)
   select=select[rank<=8][:256]
  z=h.reshape(-1,48)[select].float()
  self.decov_stats=dict(samples=len(select),trace=0.,offdiag_ratio=0.,main_loss=0.,auxiliary_loss=0.)
  if len(select)<8:return z.sum()*0
  with torch.autocast(device_type=z.device.type,enabled=False):
   centered=z-z.mean(0,keepdim=True)
   covariance=centered.T@centered/len(select)
   diagonal=covariance.diagonal();trace=diagonal.sum()
   offdiag=covariance-torch.diag_embed(diagonal)
   ratio=offdiag.square().sum()/(trace.square()+1e-8)
  self.decov_stats.update(trace=float(trace.detach()),offdiag_ratio=float(ratio.detach()))
  coefficient=.01/3 if self.decov_mode=='decorrelation' else 0.
  auxiliary=coefficient*ratio
  self.decov_stats['auxiliary_loss']=float(auxiliary.detach())
  return auxiliary
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(decov_mode=self.decov_mode,added_parameters=0,decov_feature='actual final fresh48 readout input',
   decov_sampling='CONT core row%32/dedup/hash/max8perwell/max256/min8',
   decov_formula='sum(offdiag(C)^2)/(trace(C)^2+1e-8), C=centeredH.T@centeredH/N',
   decov_coefficient=.01/3 if self.decov_mode=='decorrelation' else 0.,
   denominator_detached=False,no_label_distance_or_prediction_pairs=True,
   original154_only=True,no336_or358_trained_weights=True,one_joint_PERM_phase=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

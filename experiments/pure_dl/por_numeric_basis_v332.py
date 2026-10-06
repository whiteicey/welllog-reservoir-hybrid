"""Fresh joint POR backbone/readout with residual nonlinear numeric curve tokens."""
import torch
from torch import nn
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_numeric_basis_v332_20261002'
OUT=ROOT/'outputs/pure_dl_por_numeric_basis_v332_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('rbf','cumulative')}
class NumericBasis(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.register_buffer('centers',torch.zeros(9,8));self.register_buffer('spacing',torch.ones(9,1));self.register_buffer('enabled',torch.ones(9,1))
  self.projection=nn.Linear(8,32,bias=False);nn.init.zeros_(self.projection.weight)
 def forward(self,curves):
  with torch.autocast(device_type=curves.device.type,enabled=False):
   distances=(curves.float()-self.centers)/self.spacing
   phi=torch.exp(-distances.square()) if self.mode=='rbf' else .5*(1+torch.tanh(distances))
   return self.projection(phi*self.enabled)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'full');self.numeric_basis_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261332);self.curve_interaction.numeric_basis=NumericBasis(mode)
  self.curve_interaction.numeric.register_forward_hook(self.add_numeric_basis)
 def add_numeric_basis(self,module,args,h):return h+self.curve_interaction.numeric_basis(args[0]).to(h.dtype)
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  with torch.no_grad():
   values=data['x'][data['train_raw']].index_select(-1,self.curve_indices).float()
   bounds=torch.quantile(values,torch.tensor([.05,.95],device=values.device),dim=0)
   near_constant=bounds[1]-bounds[0]<1e-6
   bounds=torch.where(near_constant[None],torch.stack((values.amin(0),values.amax(0))),bounds)
   enabled=(bounds[1]-bounds[0]>=1e-6).float()
   spacing=((bounds[1]-bounds[0])/7).clamp_min(1e-3)
   centers=bounds[0,:,None]+torch.arange(8,device=values.device)*spacing[:,None]
   self.curve_interaction.numeric_basis.centers.copy_(centers);self.curve_interaction.numeric_basis.spacing.copy_(spacing[:,None]);self.curve_interaction.numeric_basis.enabled.copy_(enabled[:,None])
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==76848
  info.update(new_active_parameters=n,active_parameters=n,new_numeric_basis_parameters=256,numeric_basis_mode=self.numeric_basis_mode,numeric_basis_centers=centers.cpu().tolist(),numeric_basis_spacing=spacing.cpu().tolist(),basis_statistics_from64_training_input_rows_only=True,basis_enabled=enabled.cpu().tolist(),near_constant_quantile_fallback_train_full_range=True,fullconstant_basis_disabled=True,
   original_linear_numeric_path_retained=True,numeric_residual_zero_initial=True,new_curve_interaction_parameters=19280,
   all_fresh_backbone_curve_attention_readout_numeric_joint=True,no302_or_other_trained_repair_weights=True,only_original151_checkpoint=True,
   primary_control='Gaussian local versus smooth cumulative tanh, same256 trainable projection/centers/spacing but basis function families differ',same_initial_function_as_fresh302=True,
   first_step_basis_gradient_delayed_by_zero_parent_bridge=True,no_released_weights_or_labels_in_basis=True,not_first_numeric_embedding_or_Gaussian_class=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

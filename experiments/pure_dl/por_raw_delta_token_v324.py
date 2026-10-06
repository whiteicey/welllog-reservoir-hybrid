"""Fresh POR raw/normalization-displacement tokens, matched adaptive/reference."""
import torch
from torch import nn
from por_window_normalization_v322 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Normalized
from por_curve_scope_v302 import Network as CurveParent
STUDY=ROOT/'results/deep_learning/por_raw_delta_token_v324_20261002'
OUT=ROOT/'outputs/pure_dl_por_raw_delta_token_v324_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('adaptive','reference')}
class Network(Normalized):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,mode)
  old=self.curve_interaction.numeric
  with torch.random.fork_rng(devices=[]):replacement=nn.Linear(2,32)
  with torch.no_grad():
   replacement.weight.copy_(old.weight.repeat(1,2));replacement.bias.copy_(old.bias)
  self.curve_interaction.numeric=replacement
  self.curve_interaction.forward=self.raw_delta_tokens
 def raw_delta_tokens(self,curves):
  normalized=self.window_norm(curves,self.current_valid)
  pair=torch.stack((curves,normalized-curves),dim=-1)
  interaction=self.curve_interaction;shape=curves.shape[:2]
  tokens=interaction.numeric(pair.reshape(-1,9,2))+interaction.identity[None]
  for block in interaction.blocks:tokens=block(tokens,None)
  return interaction.bridge(tokens.mean(1)).reshape(*shape,48)
 def initialize_priors(self,data):
  info=CurveParent.initialize_priors(self,data)
  count=sum(p.numel() for p in self.parameters() if p.requires_grad);assert count==76786
  info.update(new_active_parameters=count,active_parameters=count,new_interaction_parameters=19056,new_normalizer_parameters=162,
   normalization_mode=self.normalization_mode,initial_normalizer_identity=True,raw_delta_numeric_inputs=2,
   both_numeric_weight_columns_copy_original=True,added_numeric_parameters=32,
   no302_or322_trained_weights_loaded=True,initial_predictor_preserved_by_zero_bridge_and_alpha=True,
   tokenizer_historical_comparison_not_capacity_matched=True,original206_plus63_dense_branch_unchanged=True,
   primary_control='same dual numeric tokenizer and parameters, actual valid window versus train input reference')
  return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)

"""Fresh jointly trained POR encoder/readout; FFN hidden dropout only."""
from collections import OrderedDict
from torch import nn
from por_coupling_conditional_rps_v341 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_ffn_dropout_v343_20261002'
OUT=ROOT/'outputs/pure_dl_por_ffn_dropout_v343_fold5'
STAGES={'dropout':('POR',256,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'conditional')
  for block in self.curve_interaction.blocks:
   old=block.ff
   block.ff=nn.Sequential(OrderedDict([('0',old[0]),('1',old[1]),('dropout',nn.Dropout(.1)),('2',old[2])]))
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(FFN_hidden_dropout=.1,dropout_blocks=2,attention_dropout=0.,dropout_eval_disabled=True,no_new_parameters=True,parameter_names_and_initialization_preserved=True,primary_control='341 conditionalRPS no dropout historical42',no341_trained_weights=True,original151_only=True,all_encoder_readout_jointly_trainable=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

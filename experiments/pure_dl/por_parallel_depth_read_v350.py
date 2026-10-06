"""Preserve positive temporal and sparse-reader mechanisms; parallel residual wiring."""
from por_adaptive_depth_read_v345 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent,ReaderInteraction
STUDY=ROOT/'results/deep_learning/por_parallel_depth_read_v350_20261002'
OUT=ROOT/'outputs/pure_dl_por_parallel_depth_read_v350_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('adaptive','fixed')}
class ParallelInteraction(ReaderInteraction):
 def forward(self,curves):
  b,l=curves.shape[:2];h=self.numeric(curves.reshape(-1,9,1))+self.identity[None]
  h=self.blocks[0](h,None).reshape(b,l,9,64);assert self.current_valid is not None
  # Both return identity plus zero-initialized residual. Avoid double counting h.
  h=(self.reader(h,self.current_valid)+self.temporal(h,self.current_valid)-h).reshape(-1,9,64)
  h=self.blocks[1](h,None)
  return self.bridge(h.mean(1)).reshape(b,l,48)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,mode)
  # Same objects and initialization; only their forward connection changes.
  self.curve_interaction.__class__=ParallelInteraction
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(depth_reader_connection='parallel residuals from same firstcurveblock features',
              no_extra_parameters_over345=True,original_temporal_and_sparse_reader_retained=True,
              primary_historical_control='345 adaptive serial, same136626params/init, nonconcurrent',
              secondary_matched_control='350fixed parallel, samephysicalparams/init, roles/effectivecapacity differs',
              no345_trained_weights=True,only_original151_checkpoint=True,all_new_encoder_reader_readout_joint=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

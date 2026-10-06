"""Transfer terminal-positive dynamic amplitude spectrum into fresh POR joint encoder."""
import torch
from por_coupling_conditional_rps_v341 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from sw_dynamic_cross_spectrum_v351 import CrossSpectrum
STUDY=ROOT/'results/deep_learning/por_dynamic_cross_spectrum_v353_20261002'
OUT=ROOT/'outputs/pure_dl_por_dynamic_cross_spectrum_v353_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('aligned','permuted')}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'conditional');self.spectrum_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261351);self.cross_spectrum=CrossSpectrum(mode)
  self.current_spectrum_correction=None
  # Existing curve-token/depth correction hook remains; append spectrum before raw branches.
  self.stem.register_forward_hook(self.add_spectrum)
 def add_spectrum(self,module,args,h):
  assert self.current_spectrum_correction is not None
  return h+self.current_spectrum_correction.to(h.dtype)
 def groups(self):return super().groups()+[dict(params=list(self.cross_spectrum.parameters()),lr=3e-4,name='POR_dynamic_cross_spectrum_backbone')]
 def initialize_priors(self,data):
  spectrum=self.cross_spectrum;del self.cross_spectrum
  try:info=super().initialize_priors(data)
  finally:self.cross_spectrum=spectrum
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in spectrum.parameters());assert n==135304 and extra==552
  info.update(active_parameters=n,new_active_parameters=n,new_spectrum_parameters=extra,spectrum_mode=self.spectrum_mode,
   only_original151_checkpoint=True,no341_or351_trained_weights=True,positive351_amplitude_code_only=True,
   positivePOR_curve_depth_conditionalRPS_retained=True,all_fresh_encoder_readout_spectrum_joint=True,
   original206_relative_input_unchanged=True,curve_indices=self.curve_indices.tolist(),POR_window256_not_SW1024=True,
   spectrumFFT512=True,target_phase_preserved=True,DC_correction_zero=True,constant_cutoff=1e-4,
   spectrum_residual_filtered_minus_original=True,zero_9to48_bridge=True,one_POR_added_phase=True,
   primary_control='aligned vs same fixed nonDC donorfrequency permutation, same135304 physicalparams/init; also changes smoothness/effective functionclass',
   historical_control='341samepositivebody without552spectrum parameters, nonconcurrent/capacitydiffers')
  return info
 def forward(self,x,relative,valid,**kwargs):
  assert x.shape[1]==256
  self.current_spectrum_correction=self.cross_spectrum(x.index_select(-1,self.curve_indices),valid)
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_spectrum_correction=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)

"""Fresh SW joint encoder/readout plus phase-preserving dynamic cross-spectrum residual."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
from sw_raw_well_condition_v297 import Network as FilmParent
STUDY=ROOT/'results/deep_learning/sw_dynamic_cross_spectrum_v351_20261002'
OUT=ROOT/'outputs/pure_dl_sw_dynamic_cross_spectrum_v351_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('aligned','permuted')}
class CrossSpectrum(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.mixing=nn.Parameter(torch.zeros(9,8));self.bridge=nn.Linear(9,48)
  nn.init.zeros_(self.bridge.weight);nn.init.zeros_(self.bridge.bias)
  donors=torch.stack([torch.tensor([j for j in range(9) if j!=i]) for i in range(9)])
  self.register_buffer('donors',donors)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261351);self.register_buffer('donor_permutation',torch.randperm(256))
 def forward(self,curves,valid):
  with torch.autocast(device_type=curves.device.type,enabled=False):
   x=curves.float();v=valid.unsqueeze(-1).float();n=v.sum(1,keepdim=True);mu=(x*v).sum(1,keepdim=True)/n.clamp_min(1)
   z=(x-mu)*v;sd=(z.square().sum(1,keepdim=True)/n.clamp_min(1)).sqrt();eligible=(n>=2)&(sd>1e-4)
   z=z*eligible;spec=torch.fft.rfft(z,n=512,dim=1,norm='ortho')
   amp=torch.log1p(spec.abs());nonDC=amp[:,1:];mean=nonDC.mean(1,keepdim=True);std=nonDC.std(1,keepdim=True,correction=0)
   donor=((nonDC-mean)/std.clamp_min(.1)).clamp(-4,4)*eligible
   if self.mode=='permuted':donor=donor.index_select(1,self.donor_permutation)
   weights=self.mixing.softmax(-1);w=torch.zeros(9,9,device=x.device,dtype=torch.float32).scatter(1,self.donors,weights)
   gate=.5*torch.tanh(torch.einsum('blj,ij->bli',donor,w))
   gate=F.pad(gate,(0,0,1,0));residual=torch.fft.irfft(spec*gate,n=512,dim=1,norm='ortho')[:,:256]
   delta=.25*torch.tanh(self.bridge(residual));return delta*v
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth');self.spectrum_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261351);self.cross_spectrum=CrossSpectrum(mode)
  self.current_spectrum_correction=None
  self.stem.register_forward_hook(self.add_spectrum)
 def add_spectrum(self,module,args,h):
  assert self.current_spectrum_correction is not None
  return h+self.current_spectrum_correction.to(h.dtype)
 def groups(self):return super().groups()+[dict(params=list(self.cross_spectrum.parameters()),lr=3e-4,name='SW_dynamic_cross_spectrum_backbone')]
 def initialize_priors(self,data):
  info=FilmParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in self.cross_spectrum.parameters());assert n==92841 and extra==552
  info.update(active_parameters=n,new_active_parameters=n,new_spectrum_parameters=extra,new_latent_fusion_parameters=21600,new_morphology_parameters=5424,morphology_mode='depth',spectrum_mode=self.spectrum_mode,
   original104_only=True,no326_or_other_trained_repair_weights=True,all_fresh_encoder_density_state_film_latent_morph_spectrum_joint=True,
   own_norm128_repeat3_input=True,central256_in_original1024=True,spectrum_FFT=512,spectrum_normalization='ortho FP32',valid_curve_centering=True,
   constant_curve_cutoff=1e-4,amplitude='nonDC log1p abs normalized percurve clip4 stdfloor.1',off_diagonal_row_softmax72=True,multiplicative_filter_bound=.5,DC_correction_zero=True,target_complex_phase_preserved=True,
   spectral_residual_filtered_minus_original=True,zero_9to48_bridge=True,bridge_bound=.25,donor_permutation_seed=20261351,
   permutation_moves_donor_frequency_only=True,physical_parameter_matched_not_effective_function_capacity=True,
   primary_control='aligned vs fixed permuted nonDC donor frequency; also breaks spectral smoothness, not pure causality',
   fusion_position='stem after FiLM and ownlatent before three branches',one_SW_added_phase=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  lo=x.shape[1]//2-128;hi=lo+256
  z=self.cross_spectrum(x[:,lo:hi].index_select(-1,self.curve_indices),valid[:,lo:hi])
  self.current_spectrum_correction=F.pad(z,(0,0,lo,x.shape[1]-hi))
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_spectrum_correction=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)

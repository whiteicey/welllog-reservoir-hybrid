"""Add finite smoothed cross-spectrum coherency description, no extra parameters."""
import torch
from torch.nn import functional as F
from sw_dynamic_cross_spectrum_v351 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent,CrossSpectrum
STUDY=ROOT/'results/deep_learning/sw_phase_coherency_v352_20261002'
OUT=ROOT/'outputs/pure_dl_sw_phase_coherency_v352_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('phase','magnitude')}
class CoherencySpectrum(CrossSpectrum):
 @staticmethod
 def smooth(v):
  shape=v.shape;z=v.movedim(1,-1).reshape(-1,shape[1])
  z=F.avg_pool1d(F.pad(z[:,None],(2,2),mode='reflect'),5,stride=1).squeeze(1)
  return z.reshape(shape[0],*shape[2:],shape[1]).movedim(-1,1)
 def forward(self,curves,valid):
  with torch.autocast(device_type=curves.device.type,enabled=False):
   x=curves.float();v=valid.unsqueeze(-1).float();n=v.sum(1,keepdim=True);mu=(x*v).sum(1,keepdim=True)/n.clamp_min(1)
   z=(x-mu)*v;sd=(z.square().sum(1,keepdim=True)/n.clamp_min(1)).sqrt();eligible=(n>=2)&(sd>1e-4)
   z=z*eligible;spec=torch.fft.rfft(z,n=512,dim=1,norm='ortho');sp=spec[:,1:]
   amp=torch.log1p(sp.abs());mean=amp.mean(1,keepdim=True);std=amp.std(1,keepdim=True,correction=0)
   donor=((amp-mean)/std.clamp_min(.1)).clamp(-4,4)*eligible
   cross=sp.unsqueeze(-1)*sp[:,:,self.donors].conj()
   cross=torch.complex(self.smooth(cross.real),self.smooth(cross.imag))
   power=self.smooth(sp.abs().square());eps=1e-6*power.mean(1,keepdim=True)+1e-12;power=torch.maximum(power,eps)
   denominator=(power.unsqueeze(-1)*power[:,:,self.donors]).sqrt()
   rho=cross/denominator;rho=rho/rho.abs().clamp_min(1)
   rho=rho*(eligible.unsqueeze(-1)&eligible[:,:,self.donors])
   descriptor=rho.real if self.mode=='phase' else rho.abs()
   features=donor[:,:,self.donors]+.25*descriptor
   gate=.5*torch.tanh((features*self.mixing.softmax(-1)[None,None]).sum(-1))
   gate=F.pad(gate,(0,0,1,0));residual=torch.fft.irfft(spec*gate,n=512,dim=1,norm='ortho')[:,:256]
   return .25*torch.tanh(self.bridge(residual))*v
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,mode);self.cross_spectrum.__class__=CoherencySpectrum
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(no351_trained_weights=True,new_parameters_over351=0,coherency_frequency_average=5,frequency_boundary_padding='reflect2 nonDC only',
   coherency_epsilon='1e-6 percurve mean smoothed power+1e-12',coherency_unit_disc_rescaling=True,
   coherency_descriptor='real rho' if self.spectrum_mode=='phase' else 'abs rho',coherency_contribution=.25,
   no_donor_permutation_in_this_pair=True,unused_inherited_permutation_buffer=True,not_Welch_estimate=True,
   primary_control='real smoothed rho vs magnitude smoothed rho; same92841params/init but effective functions differ',
   target_phase_preserved_not_donor_phase_transferred=True,only_original104_checkpoint=True,all_fresh_encoder_readout_coherency_joint=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

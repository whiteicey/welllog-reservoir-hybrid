"""Matched local/frequency residual at original SW point resolution.

No old trained SW or source feature encoder. Source pretraining remains counted
for the shared POR/PERM and training-only Q preparation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,CompactSW,load_data as original_data

STUDY=ROOT/'results/deep_learning/compact_frequency_v104_20261001'
OUT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('local','spectral')}

class DepthFilter(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.norm=nn.LayerNorm(96)
        # 33 real modes including DC, 32 imaginary modes excluding DC = 65.
        # Exactly the same coefficients as a real 65-tap depthwise convolution.
        self.coefficients=nn.Parameter(torch.zeros(96,65))

    def forward(self,h,valid):
        z=self.norm(h).float().masked_fill(~valid[...,None],0).transpose(1,2)
        if self.mode=='local':
            value=F.conv1d(z,self.coefficients[:,None,:],padding=32,groups=96)
        else:
            with torch.autocast(device_type=h.device.type,enabled=False):
                length=h.shape[1];spectrum=torch.fft.rfft(z,n=2*length,dim=-1,norm='ortho')
                imaginary=torch.cat((self.coefficients.new_zeros(96,1),self.coefficients[:,33:]),1)
                weights=torch.complex(self.coefficients[:,:33],imaginary)
                low=spectrum[...,:33]*weights[None]
                filtered=torch.cat((low,torch.zeros_like(spectrum[...,33:])),dim=-1)
                # Zero-pad to 2L, crop original points: no periodic opposite-edge input.
                value=torch.fft.irfft(filtered,n=2*length,dim=-1,norm='ortho')[...,:length]
        return (h+value.transpose(1,2).to(h.dtype)).masked_fill(~valid[...,None],0)

class Network(CompactSW):
    def __init__(self,metadata,mode):
        super().__init__(metadata);self.filter_mode=mode
        self.body.local.append(DepthFilter(mode))

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(metadata,stage)

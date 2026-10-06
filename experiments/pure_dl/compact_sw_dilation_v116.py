"""Fixed-capacity SW depth-filter dilation; same zero initialized coefficients."""
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,CompactSW,load_data as original_data
from compact_frequency_v104 import DepthFilter as OriginalFilter
STUDY=ROOT/'results/deep_learning/compact_sw_dilation_v116_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_dilation_v116_fold5'
STAGES={n:('SW',1024,True,1.,3e-4) for n in ('dilation2','dilation4')}
class DepthFilter(OriginalFilter):
    def __init__(self,dilation):super().__init__('local');self.dilation=dilation
    def forward(self,h,valid):
        z=self.norm(h).float().masked_fill(~valid[...,None],0).transpose(1,2)
        value=F.conv1d(z,self.coefficients[:,None,:],padding=32*self.dilation,dilation=self.dilation,groups=96)
        return (h+value.transpose(1,2).to(h.dtype)).masked_fill(~valid[...,None],0)
class Network(CompactSW):
    def __init__(self,metadata,mode):
        super().__init__(metadata);self.filter_mode=mode;self.body.local.append(DepthFilter(int(mode[-1])))
def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(metadata,stage)

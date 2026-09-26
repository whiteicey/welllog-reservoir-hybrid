"""Input-only bidirectional normalized exponential context; not parameter EMA."""
import torch
from torch import nn
from patch_hierarchy_sequence import Network as Baseline,PatchStem

def normalized_filter(x,valid,decay):
    length=x.shape[-1];nfft=2*length
    mask=valid[:,None].float().expand_as(x)
    kernel=(1-decay[:,None])*decay[:,None].pow(torch.arange(length,device=x.device))
    stacked=torch.cat([x.float()*mask,mask],0)
    result=torch.fft.irfft(torch.fft.rfft(stacked,n=nfft)*torch.fft.rfft(kernel,n=nfft)[None],n=nfft)[...,:length]
    numerator,denominator=result.chunk(2,0)
    return (numerator/denominator.clamp_min(1e-6)).masked_fill(~valid[:,None],0)

class ExponentialContext(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(64);self.input=nn.Conv1d(64,16,1)
        tau=torch.logspace(torch.log10(torch.tensor(2.)).item(),torch.log10(torch.tensor(64.)).item(),16)
        self.rate=nn.Parameter(torch.logit(torch.exp(-1/tau)))
        self.output=nn.Conv1d(16,64,1,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,x,valid):
        z=self.input(self.norm(x.transpose(1,2)).transpose(1,2)).masked_fill(~valid[:,None],0)
        decay=self.rate.float().sigmoid().clamp(.01,.995)
        context=.5*(normalized_filter(z,valid,decay)+normalized_filter(z.flip(-1),valid.flip(-1),decay).flip(-1))
        return self.output((context-z.float()).to(z.dtype)).masked_fill(~valid[:,None],0)

class ContextStem(PatchStem):
    def __init__(self,old):
        super().__init__(old);self.load_state_dict(old.state_dict());self.context=ExponentialContext()
    def forward(self,x):
        result=super().forward(x)
        return result+self.context(result,x[:,-2,:]>0)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.stem=ContextStem(self.stem)

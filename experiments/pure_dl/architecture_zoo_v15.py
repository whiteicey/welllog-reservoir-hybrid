"""Three different end-to-end inductive biases for private SW prediction.

All read the same207 features and output the same default+256bin distribution.
They are task-specific adaptations, not complete DCN/Perceiver/S4D reproductions.
No previous prediction, pretrained weight, label memory, or other-target model.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_scratch_v10 import Network as BaseNetwork, TemporalBlock

ARCHITECTURES = ('adaptive', 'latent', 'state_space')
DESCRIPTIONS = {
    'adaptive': '4 content-conditioned fractional-depth sampling blocks;5 taps;spacing1,2,4,8;bounded offsets and learned tap weights;full resolution',
    'latent': 'local full-resolution path plus16 learned latent tokens:read256 points,2 self-attention blocks,write back256 point queries;4 heads,width64',
    'state_space': '4 bidirectional diagonal exponential-memory blocks with learned stable timescales2..128,content output gates,FFNs;FFT implements zero-state recurrence,no circular wrap',
}


def clean(x, valid):
    return x.masked_fill(~valid[..., None], 0)


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, h, valid):
        return clean(h + self.net(clean(self.norm(h), valid)), valid)


class AdaptiveSampling(nn.Module):
    """Per-query, differentiable linear sampling; no coordinates cross the window."""
    def __init__(self, width, spacing):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.coordinates = nn.Linear(width, 10)
        nn.init.zeros_(self.coordinates.weight)
        nn.init.zeros_(self.coordinates.bias)
        self.value = nn.Linear(width, width)
        self.project = nn.Linear(width, width)
        self.ffn = FeedForward(width)
        self.spacing = float(spacing)
        self.register_buffer('base_offsets', torch.arange(-2, 3, dtype=torch.float32)*spacing)
        self.freeze_offsets_for_diagnostic = False

    @staticmethod
    def sample(values, valid, position):
        b, length, width = values.shape
        lo = position.floor().long(); hi = lo+1
        fraction = (position-lo).unsqueeze(-1)
        def gather(index):
            inrange = (index >= 0) & (index < length)
            at = index.clamp(0, length-1).reshape(b, -1)
            data = values.gather(1, at[..., None].expand(-1, -1, width))
            mask = valid.gather(1, at).reshape_as(index) & inrange
            return data.reshape(*index.shape, width).masked_fill(~mask[..., None], 0)
        return gather(lo)*(1-fraction) + gather(hi)*fraction

    def forward(self, h, valid):
        x = clean(self.norm(h), valid)
        control = self.coordinates(x).float()
        offset = control[..., :5].tanh()*(2*self.spacing)
        if self.freeze_offsets_for_diagnostic:
            offset = torch.zeros_like(offset)
        position = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[None, :, None]
        position = position+self.base_offsets+offset
        values = clean(self.value(x), valid)
        sampled = self.sample(values, valid, position)
        weights = control[..., 5:].softmax(-1)
        message = (weights[..., None]*sampled).sum(-2)
        h = clean(h+self.project(message.to(h.dtype)), valid)
        return self.ffn(h, valid)


class CrossAttention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.width=width; self.heads=heads; self.dim=width//heads
        self.q=nn.Linear(width,width); self.kv=nn.Linear(width,2*width); self.out=nn.Linear(width,width)
        self.qnorm=nn.LayerNorm(width); self.knorm=nn.LayerNorm(width)

    def forward(self, query, memory, query_valid, memory_valid):
        b,n,_=query.shape;length=memory.shape[1]
        q=self.q(clean(self.qnorm(query),query_valid)).reshape(b,n,self.heads,self.dim).transpose(1,2)
        kv=self.kv(clean(self.knorm(memory),memory_valid)).reshape(b,length,2,self.heads,self.dim)
        k=kv[:,:,0].transpose(1,2);v=kv[:,:,1].transpose(1,2)
        # One artificial zero key only for empty rows; outputs are masked below.
        safe=memory_valid.clone();empty=~safe.any(1);safe[empty,0]=True
        k=k.masked_fill(~memory_valid[:,None,:,None],0);v=v.masked_fill(~memory_valid[:,None,:,None],0)
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None,None,:],dropout_p=0.)
        z=z.transpose(1,2).reshape(b,n,self.width)
        return self.out(z).masked_fill(~(query_valid&~empty[:,None])[...,None],0)


class LatentMemory(nn.Module):
    def __init__(self,width=64,count=16):
        super().__init__()
        self.latents=nn.Parameter(torch.randn(count,width)*.02)
        self.read=CrossAttention(width);self.write=CrossAttention(width)
        self.attn=nn.ModuleList([CrossAttention(width) for _ in range(2)])
        self.ffn=nn.ModuleList([FeedForward(width) for _ in range(2)])
        self.output_ffn=FeedForward(width)
        self.disable_global_for_diagnostic=False

    def forward(self,h,valid):
        if self.disable_global_for_diagnostic:
            return self.output_ffn(h,valid)
        b=h.shape[0]
        z=self.latents[None].expand(b,-1,-1)
        latent_valid=valid.any(1)[:,None].expand(b,len(self.latents))
        z=clean(z+self.read(z,h,latent_valid,valid),latent_valid)
        for attn,ffn in zip(self.attn,self.ffn):
            z=ffn(clean(z+attn(z,z,latent_valid,latent_valid),latent_valid),latent_valid)
        return self.output_ffn(clean(h+self.write(h,z,valid,latent_valid),valid),valid)


class BidirectionalStateSpace(nn.Module):
    """Real diagonal one-state/channel recurrence, accelerated by linear convolution.

    s[t]=a*s[t-1]+(1-a)*u[t]; learn a via a=exp(-1/tau), tau>1.
    Different channels provide a bank of timescales. Content-dependent output
    gating is NOT input-dependent transition A; this is not a Mamba replica.
    """
    def __init__(self,width=64):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.input=nn.Linear(width,2*width)
        self.output=nn.Linear(width,width)
        tau=torch.logspace(math.log10(2),math.log10(128),width)
        t=tau-1
        inverse_softplus=t+torch.log(-torch.expm1(-t))
        self.raw_tau=nn.Parameter(torch.stack([inverse_softplus,inverse_softplus.flip(0)]))
        self.skip=nn.Parameter(torch.ones(width))
        self.direction=nn.Parameter(torch.zeros(width))
        self.ffn=FeedForward(width)
        self.disable_memory_for_diagnostic=False

    def convolve(self,u,direction):
        # Float32 FFT avoids mixed-precision restrictions; n>=2L-1 prevents wrap.
        length=u.shape[-1];tau=F.softplus(self.raw_tau[direction].float())+1
        age=torch.arange(length,device=u.device,dtype=torch.float32)
        kernel=-torch.expm1(-1/tau[:,None])*torch.exp(-age[None]/tau[:,None])
        result=torch.fft.irfft(torch.fft.rfft(u.float(),n=2*length)*torch.fft.rfft(kernel,n=2*length)[None],n=2*length)
        return result[...,:length]

    def forward(self,h,valid):
        x=clean(self.norm(h),valid)
        raw,gate=self.input(x).chunk(2,-1)
        u=clean(F.silu(raw),valid).transpose(1,2)
        with torch.autocast(device_type=h.device.type,enabled=False):
            if self.disable_memory_for_diagnostic:
                memory=torch.zeros_like(u.float())
            else:
                forward=self.convolve(u,0)
                backward=self.convolve(u.flip(-1),1).flip(-1)
                mix=self.direction.sigmoid()[None,:,None]
                memory=mix*forward+(1-mix)*backward
            z=(memory+self.skip[None,:,None]*u.float()).transpose(1,2)
        h=clean(h+self.output(z.to(h.dtype)*F.silu(gate)),valid)
        return self.ffn(h,valid)


class Network(BaseNetwork):
    def __init__(self,architecture,centers,target_median,target_scale):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        # Reuse only the common interface and identical stem/output initialization.
        super().__init__('tcn',centers,target_median,target_scale)
        self.architecture=architecture
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            if architecture=='adaptive':
                self.blocks=nn.ModuleList([AdaptiveSampling(64,d) for d in (1,2,4,8)])
            elif architecture=='latent':
                self.blocks=nn.ModuleList([TemporalBlock(64,1),LatentMemory(64,16),TemporalBlock(64,2)])
            else:
                self.blocks=nn.ModuleList([BidirectionalStateSpace(64) for _ in range(4)])

    @torch.no_grad()
    def mechanism_statistics(self):
        out={}
        if self.architecture=='adaptive':
            out['offset_predictor_weight_norm']=[float(x.coordinates.weight[:5].float().norm()) for x in self.blocks]
        elif self.architecture=='latent':
            out['latent_parameter_norm']=float(self.blocks[1].latents.float().norm())
        else:
            taus=torch.cat([(F.softplus(b.raw_tau)+1).flatten() for b in self.blocks])
            out.update(learned_tau_min=float(taus.min()),learned_tau_max=float(taus.max()),learned_tau_median=float(taus.median()))
        return out

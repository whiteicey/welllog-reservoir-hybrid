"""Private POR multiresolution backbone with a gated latent side path.

The two modes have identical parameters/initial states. Only three attention
connectivity masks differ: regional versus full-window communication. All
input statistics and POR distribution semantics remain those used in v18.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_scratch_v10 import TemporalBlock
from architecture_zoo_v15 import clean, FeedForward
from por_architecture_v18 import Network as PORBase

MODES = ('regional', 'global')


def downsample(h, valid):
    """Masked pairwise average; invalid positions never contribute."""
    b, length, width = h.shape
    if length % 2:
        raise ValueError('Pyramid requires an even sequence length')
    mask = valid.reshape(b, length//2, 2)
    counts = mask.sum(-1)
    values = clean(h, valid).reshape(b, length//2, 2, width).sum(2)
    result = values / counts.clamp_min(1)[..., None].to(values.dtype)
    return clean(result, counts > 0), counts > 0


class MaskedAttention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads, self.dim = heads, width//heads
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv, self.out = nn.Linear(width,width), nn.Linear(width,2*width), nn.Linear(width,width)

    def forward(self, query, memory, query_valid, memory_valid, allowed):
        b,n,width=query.shape; length=memory.shape[1]
        q=self.q(clean(self.qnorm(query),query_valid)).reshape(b,n,self.heads,self.dim).transpose(1,2)
        kv=self.kv(clean(self.knorm(memory),memory_valid)).reshape(b,length,2,self.heads,self.dim)
        k=kv[:,:,0].transpose(1,2).masked_fill(~memory_valid[:,None,:,None],0)
        v=kv[:,:,1].transpose(1,2).masked_fill(~memory_valid[:,None,:,None],0)
        mask=memory_valid[:,None,:] & allowed[None,:,:]
        available=mask.any(-1)
        safe=mask.clone(); safe[:,:,0] |= ~available
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None,:,:],dropout_p=0.)
        z=z.transpose(1,2).reshape(b,n,width)
        return self.out(z).masked_fill(~(query_valid & available)[...,None],0)


class GatedContext(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode=mode
        self.project_in=nn.Linear(width,64)
        self.latents=nn.Parameter(torch.randn(16,64)*.02)
        self.read=MaskedAttention(); self.write=MaskedAttention()
        self.process=nn.ModuleList([MaskedAttention(),MaskedAttention()])
        self.ffn=nn.ModuleList([FeedForward(64),FeedForward(64)])
        self.project_out=nn.Linear(64,width)
        self.gate=nn.Linear(width,width)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,-2.)
        self.norm=nn.LayerNorm(width)
        self.disabled_for_diagnostic=False

    def masks(self,length,device):
        slots=torch.arange(16,device=device,dtype=torch.float32)
        centers=(slots+.5)*(length/16)-.5
        points=torch.arange(length,device=device,dtype=torch.float32)
        if self.mode=='global':
            return (torch.ones(16,length,device=device,dtype=torch.bool),
                    torch.ones(16,16,device=device,dtype=torch.bool),
                    torch.ones(length,16,device=device,dtype=torch.bool))
        read=(centers[:,None]-points[None,:]).abs() <= length/8
        process=(slots[:,None]-slots[None,:]).abs() <= 1
        write=(points[:,None]-centers[None,:]).abs() <= length/8
        return read,process,write

    def forward(self,h,valid):
        if self.disabled_for_diagnostic:
            return h
        x=clean(self.project_in(clean(self.norm(h),valid)),valid)
        readmask,processmask,writemask=self.masks(h.shape[1],h.device)
        slot_valid=(valid[:,None,:] & readmask[None,:,:]).any(-1)
        z=self.latents[None].expand(h.shape[0],-1,-1)
        z=clean(z+self.read(z,x,slot_valid,valid,readmask),slot_valid)
        for attn,ffn in zip(self.process,self.ffn):
            z=ffn(clean(z+attn(z,z,slot_valid,slot_valid,processmask),slot_valid),slot_valid)
        message=self.write(x,z,valid,slot_valid,writemask)
        gate=self.gate(clean(self.norm(h),valid)).sigmoid()
        return clean(h+gate*self.project_out(message),valid)


class Network(PORBase):
    def __init__(self,mode,metadata):
        nn.Module.__init__(self)
        if mode not in MODES: raise ValueError(mode)
        self.mode=mode; self.width=96; w=self.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            self.stem=nn.Linear(207,w)
            self.blocks=nn.ModuleDict({
                'encoder0':nn.ModuleList([TemporalBlock(w,1),TemporalBlock(w,2)]),
                'encoder1':nn.ModuleList([TemporalBlock(w,1),TemporalBlock(w,2)]),
                'encoder2':nn.ModuleList([TemporalBlock(w,1),TemporalBlock(w,2)]),
                'context':GatedContext(w,mode),
                'fuse1':nn.Linear(2*w,w), 'decode1':TemporalBlock(w,1),
                'fuse0':nn.Linear(2*w,w), 'decode0':TemporalBlock(w,1)})
            self.norm=nn.LayerNorm(w)
            self.output=nn.Linear(w,25)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.register_buffer('centers',torch.as_tensor(metadata['centers'],dtype=torch.float32).clone())
        self.register_buffer('density_edges',torch.as_tensor(metadata['density_edges'],dtype=torch.float32).clone())
        self.target_median=float(metadata['target_median']); self.target_scale=float(metadata['target_scale'])
        self.metadata=metadata
        with torch.no_grad():
            rate=torch.tensor(float(metadata['training_default_rate']))
            self.output.bias[0].copy_(torch.logit(rate))
            self.output.bias[9:17].copy_(torch.as_tensor(metadata['initial_means'],dtype=torch.float32))

    def forward(self,x,relative,valid):
        h=clean(F.gelu(self.stem(clean(torch.cat((x.float(),relative[...,None].float()),-1),valid))),valid)
        for block in self.blocks['encoder0']: h=block(h,valid)
        skip0=h
        h,valid1=downsample(h,valid)
        for block in self.blocks['encoder1']: h=block(h,valid1)
        skip1=h
        h,valid2=downsample(h,valid1)
        for block in self.blocks['encoder2']: h=block(h,valid2)
        h=self.blocks['context'](h,valid2)
        up=h.repeat_interleave(2,1)
        h=clean(skip1+self.blocks['fuse1'](torch.cat((skip1,up),-1)),valid1)
        h=self.blocks['decode1'](h,valid1)
        up=h.repeat_interleave(2,1)
        h=clean(skip0+self.blocks['fuse0'](torch.cat((skip0,up),-1)),valid)
        h=self.blocks['decode0'](h,valid)
        output=clean(self.output(self.norm(h)).float(),valid)
        return output[...,0],output[...,1:].reshape(*output.shape[:2],3,8)

    @torch.no_grad()
    def mechanism_statistics(self):
        context=self.blocks['context']
        return dict(mode=self.mode,width=self.width,pyramid_lengths=[256,128,64],
            high_resolution_skip_connections=2,latent_width=64,latent_count=16,
            gate_bias_sigmoid_mean=float(context.gate.bias.sigmoid().mean()),
            gate_weight_norm=float(context.gate.weight.float().norm()),
            latent_norm=float(context.latents.float().norm()),
            head='8xStudentT_df3+default_gate',all_parameters_trainable=True)

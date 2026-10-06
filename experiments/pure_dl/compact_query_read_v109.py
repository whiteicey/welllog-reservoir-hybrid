"""Matched geometric/content queries for a bounded SW global-read residual."""
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_frequency_v104 import Network as LocalNetwork
from well_conditioned_v46 import modulate

STUDY=ROOT/'results/deep_learning/compact_query_read_v109_20261001'
OUT=ROOT/'outputs/pure_dl_compact_query_read_v109_fold5'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('geometry','content')}

class QueryRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm=nn.LayerNorm(96);self.knorm=nn.LayerNorm(96)
        self.query=nn.Linear(96,32,bias=False);self.key=nn.Linear(96,32,bias=False)
        self.value=nn.Linear(96,32,bias=False);self.output=nn.Linear(32,96,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,query,memory,valid):
        b,l,_=query.shape;t=memory.shape[1]
        q=self.query(self.qnorm(query)).reshape(b,l,4,8).transpose(1,2)
        normalized=self.knorm(memory)
        k=self.key(normalized).reshape(b,t,4,8).transpose(1,2)
        v=self.value(normalized).reshape(b,t,4,8).transpose(1,2)
        read=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        raw=self.output(read.transpose(1,2).reshape(b,l,32)).float()
        # Bound only the new residual, in fixed feature units, preserving zero init.
        return .25*torch.tanh(raw/.25)

class Network(LocalNetwork):
    def __init__(self,metadata,mode):
        super().__init__(metadata,'local');self.query_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109);self.query_read=QueryRead()

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Input-only well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1);tv=pv.any(-1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        positions=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        point_position=torch.stack((relative,positions),-1)
        loc=point_position.reshape(b,l//16,16,2);loc=(loc*pv[...,None]).sum(2)/n[...,None]
        token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        query=h if self.query_mode=='content' else F.gelu(c.token_pos(point_position))
        delta=self.query_read(query,token,tv).to(global_h.dtype).masked_fill(~valid[...,None],0)
        global_h=global_h+delta
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(b,l,1,3,8))
        return (output,y.new_zeros(()),None) if return_aux else output

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(metadata,stage)

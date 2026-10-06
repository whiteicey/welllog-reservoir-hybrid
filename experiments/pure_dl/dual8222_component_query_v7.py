"""Component-query cross-attention on a fully frozen, verified v5 predictor.

Two exact-parameter controls: center repeats each expert's current hidden token
across the SAME memory slots; local reads neighboring hidden tokens. Both use
identical initialization, relative offsets, validity masks, query descriptors,
optimizer and bounded output. Only the new query module trains; the old linear
locations also stay frozen. No external wells, model votes or label memories.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from dual8222_location_readout_v5 import Network as PreviousNetwork
from paired_detached_router_sequence import routed_merge

MODES = {'center': ('component_query_center_memory',),
         'local': ('component_query_local_memory',)}


class ComponentQueryReadout(nn.Module):
    """Features B,L,2,128; descriptors B,L,2,8,3; proposals B,L,16."""
    def __init__(self, local: bool):
        super().__init__()
        self.local = local
        self.radius = 8
        self.width = 32
        self.project = nn.Linear(128,32)
        self.expert_embedding = nn.Parameter(torch.empty(2,32))
        self.component_embedding = nn.Parameter(torch.empty(2,8,32))
        self.relative_embedding = nn.Parameter(torch.empty(17,32))
        nn.init.normal_(self.expert_embedding,std=.02)
        nn.init.normal_(self.component_embedding,std=.02)
        nn.init.normal_(self.relative_embedding,std=.02)
        self.describe = nn.Linear(3,32)
        self.token_norm = nn.LayerNorm(32)
        self.query_norm = nn.LayerNorm(32)
        self.q_proj = nn.Linear(32,32)
        self.k_proj = nn.Linear(32,32)
        self.v_proj = nn.Linear(32,32)
        self.out_proj = nn.Linear(32,32)
        self.ff_norm = nn.LayerNorm(32)
        self.ff = nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        self.output_norm = nn.LayerNorm(32)
        self.output = nn.Linear(32,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,features,descriptors,valid):
        b,l,e,d = features.shape
        if e!=2 or d!=128 or descriptors.shape!=(b,l,2,8,3):
            raise ValueError('Unexpected component-query input shapes')
        tokens = self.project(features)+self.expert_embedding[None,None]
        tokens = self.token_norm(tokens).masked_fill(~valid[:,:,None,None],0)
        query_state = self.query_norm(tokens[:,:,:,None,:]+self.component_embedding[None,None]
                                      +self.describe(descriptors))
        query_state = query_state.reshape(b,l,16,32)
        query = self.q_proj(query_state).reshape(b,l,16,2,16).transpose(2,3)
        # slots: 17 relative depth offsets x two INTERNAL expert feature tokens.
        position = torch.arange(l,device=features.device)
        offset = torch.arange(-self.radius,self.radius+1,device=features.device)
        index = position[:,None]+offset[None,:]
        in_window = (index>=0)&(index<l)
        index = index.clamp(0,l-1)
        allowed = in_window[None]&valid[:,index]
        # The center-memory control uses exactly the same masks and slot offsets.
        if self.local:
            memory = tokens[:,index]
        else:
            memory = tokens[:,:,None].expand(-1,-1,17,-1,-1)
        memory = memory+self.relative_embedding[None,None,:,None,:]
        memory = memory.reshape(b,l,34,32)
        allowed = allowed[:,:,:,None].expand(-1,-1,-1,2).reshape(b,l,34)
        # Every valid query can at least see its center. Invalid queries get one
        # dummy memory key for finite softmax and are zeroed at the module output.
        allowed = allowed&valid[:,:,None]
        dummy = torch.zeros_like(allowed);dummy[:,:,0]=True
        allowed = torch.where(valid[:,:,None],allowed,dummy)
        keys = self.k_proj(memory).reshape(b,l,34,2,16).transpose(2,3)
        values = self.v_proj(memory).reshape(b,l,34,2,16).transpose(2,3)
        with torch.autocast(device_type=features.device.type,enabled=False):
            score = query.float()@keys.float().transpose(-1,-2)/math.sqrt(16)
            score = score.masked_fill(~allowed[:,:,None,None,:],float('-inf'))
            attention = score.softmax(-1)
            gathered = (attention@values.float()).transpose(2,3).reshape(b,l,16,32)
        mixed = query_state+self.out_proj(gathered.to(query_state.dtype))
        mixed = mixed+self.ff(self.ff_norm(mixed))
        result = self.output(self.output_norm(mixed)).squeeze(-1).float()
        return result.masked_fill(~valid[:,:,None],0)


class Network(PreviousNetwork):
    def __init__(self,original,previous_checkpoint,mode):
        if mode not in MODES or previous_checkpoint['readout']['mode']!='point32':
            raise ValueError('Expected center/local mode and verified v5 checkpoint')
        super().__init__(original,previous_checkpoint['initialization_manifest'],'point32')
        self.heads.load_state_dict(previous_checkpoint['readout']['head_state_dict'],strict=True)
        self.requires_grad_(False)
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.heads[0][0].component_readout = ComponentQueryReadout(local=mode=='local')
        assert all('.component_readout.' in name for name,p in self.named_parameters() if p.requires_grad)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        if auxiliary: raise ValueError('No auxiliary objective in this readout study')
        self.captured.clear()
        prepared=[]; descriptions=[]
        with torch.no_grad():
            for member in self.backbone.members:
                member(features,relative_depth,valid,absolute_depth=absolute_depth)
            logits = self.backbone.route_logits(features,valid)
            hidden = torch.stack([self.captured[e].transpose(1,2) for e in range(2)],dim=2)
            for e in range(2):
                h=self.captured[e];heads=[self.heads[c][e] for c in range(3)]
                gate=F.conv1d(h,torch.cat([head.gate_weight for head in heads]),torch.cat([head.gate_bias for head in heads]))
                rows=[head.density_rows() for head in heads]
                raw=F.conv1d(h,torch.cat([v[0] for v in rows]),torch.cat([v[1] for v in rows]))
                raw=raw.transpose(1,2).reshape(len(h),-1,3,3,8).float()
                por=raw[:,:,0]
                old_delta=heads[0].location_residual(h,valid,por[:,:,2].transpose(1,2)).transpose(1,2)
                mean=por[:,:,1]+old_delta
                std=F.softplus(por[:,:,2])+.025
                log_weight=por[:,:,0].log_softmax(-1)
                descriptor=torch.stack([mean.clamp(-8,8),std.log().clamp(-8,8),log_weight.clamp(-8,0)],dim=-1)
                descriptions.append(descriptor)
                prepared.append((gate.transpose(1,2),raw,mean,std))
        descriptor=torch.stack(descriptions,dim=2).detach()
        proposal=self.heads[0][0].component_readout(hidden.detach(),descriptor,valid)
        outputs=[]
        for e,(gate,raw,mean,std) in enumerate(prepared):
            bound=.25*std
            delta=(bound*torch.tanh(proposal[:,:,8*e:8*(e+1)]/bound)).masked_fill(~valid[:,:,None],0)
            por=raw[:,:,0]
            updated=torch.stack([por[:,:,0],mean+delta,por[:,:,2]],dim=2)
            outputs.append((gate,torch.cat([updated.unsqueeze(2),raw[:,:,1:]],dim=2)))
        self.captured.clear()
        return routed_merge(outputs,logits)

    def export_readout(self):
        return {'format_version':1,'model_module':'dual8222_component_query_v7','mode':self.mode,
                'head_state_dict':{k:v.detach().cpu().clone() for k,v in self.heads.state_dict().items()},
                'requires_initialization_manifest':True,'new_residual_bound_fraction':.25,
                'existing_predictor_fully_frozen':True,'component_queries':16,'memory_slots':34}

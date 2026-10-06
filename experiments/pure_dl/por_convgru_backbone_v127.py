"""New private POR backbone: masked ConvNeXt blocks + packed bidirectional GRU."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from train_stage_fullcv_v97 import Evaluator as BaseEvaluator
from train_scaling_v21 import channel_objective as base_loss
STUDY=ROOT/'results/deep_learning/por_convgru_backbone_v127_20261001'
OUT=ROOT/'outputs/pure_dl_por_convgru_backbone_v127_fold5'
STAGES={'convgru':('POR',256,False,0.,6e-4)}
class ConvBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.norm=nn.LayerNorm(96)
        self.depth=nn.Conv1d(96,96,15,padding=7*dilation,dilation=dilation,groups=96)
        self.expand=nn.Linear(96,384);self.reduce=nn.Linear(192,96)
        self.scale=nn.Parameter(torch.full((96,),.01))
    def forward(self,h,valid):
        z=self.norm(h).masked_fill(~valid[...,None],0)
        z=self.depth(z.transpose(1,2)).transpose(1,2).masked_fill(~valid[...,None],0)
        a,b=self.expand(z).chunk(2,-1)
        return (h+self.scale*self.reduce(F.gelu(a)*b.sigmoid())).masked_fill(~valid[...,None],0)
class Network(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Linear(270,96)
        self.local=nn.ModuleList([ConvBlock(d) for d in (1,2,4)])
        self.context=nn.Sequential(nn.Linear(618,192),nn.GELU(),nn.Linear(192,192))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
        self.rnn_norm=nn.LayerNorm(96);self.rnn=nn.GRU(96,64,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.fuse=nn.Linear(128,96);self.rnn_scale=nn.Parameter(torch.tensor(.1))
        self.norm=nn.LayerNorm(96);self.output=nn.Linear(96,25)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def groups(self):return [dict(params=list(self.parameters()),lr=6e-4,name='new_private_POR_backbone_head')]
    @torch.no_grad()
    def initialize_priors(self,data):
        ids=data['train_raw'];d=data['default'][ids,0];y=data['labels'][ids,0][~d].float()
        rate=d.float().mean().clamp(1e-4,1-1e-4);means=torch.quantile(y,torch.linspace(.06,.94,8,device=y.device))
        self.output.bias[0]=torch.logit(rate);self.output.bias[9:17].copy_(means)
        self.output.bias[17:25].fill_(math.log(math.expm1(.5-.025)))
        return dict(random_all_network_weights=True,no_old_backbone_weights_loaded=True,own_train_label_prior_only=True,
            original_train_rows=len(ids),default_rate=float(rate),initial_locations=means.cpu().tolist(),initial_component_sd=.5,
            input_features=269,relative_depth_input=True,well_context_input_only=True,backbone='three masked depthwise ConvNeXt-GLU blocks + two-layer packed BiGRU64',
            total_recipe_training_stages=4,source_counted_for_fixedPERM_and_Q=True,GRU_float32=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        assert x.shape[-1]==269 and well_inputs is not None
        h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        inputs=well_inputs.float();stats=torch.cat((inputs.mean(1),inputs.var(1,unbiased=False).add(1e-6).sqrt(),inputs.amax(1)),1)
        gamma,beta=(.5*self.context(stats).float().tanh()).chunk(2,-1)
        h=(h*(1+gamma[:,None])+beta[:,None]).masked_fill(~valid[...,None],0)
        for block in self.local:h=block(h,valid)
        b,l,w=h.shape;lengths=valid.sum(-1);starts=valid.int().argmax(-1);pos=torch.arange(l,device=h.device)[None]
        gather=(starts[:,None]+pos).clamp_max(l-1)
        z=self.rnn_norm(h).gather(1,gather[...,None].expand(b,l,w))
        # Pack only the contiguous valid well interval; padding never enters recurrence.
        with torch.autocast('cuda',enabled=False):
            packed=nn.utils.rnn.pack_padded_sequence(z.float(),lengths.cpu(),batch_first=True,enforce_sorted=False)
            encoded,_=self.rnn(packed)
            dense,_=nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=l)
        restore=(pos-starts[:,None]).clamp_min(0)
        message=dense.gather(1,restore[...,None].expand(b,l,128)).masked_fill(~valid[...,None],0)
        h=(h+self.rnn_scale*self.fuse(message)).masked_fill(~valid[...,None],0)
        y=self.output(self.norm(h)).float()
        return y[...,:1],y[...,1:].reshape(b,l,1,3,8)
def load_data(stage):return original_data('por')
def make_model(stage,master,metadata):return Network()
def channel_objective(output,truth,default,mask,quality,target_index,mode):return base_loss(output,truth,default,mask,quality,target_index)
class Evaluator(BaseEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)

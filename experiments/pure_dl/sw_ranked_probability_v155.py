"""Keep effective SW local architecture/utility; add own joint RPS supervision."""
import math
import torch
from torch.utils.checkpoint import checkpoint
from compact_frequency_v104 import ROOT,SOURCE,Network as Parent,load_data
from sw_utility_strength_v82 import DecisionObjective as ParentObjective
from perm_ranked_probability_v154 import ranked_loss
from src.utils import DEFAULTS
STUDY=ROOT/'results/deep_learning/sw_ranked_probability_v155_20261001'
OUT=ROOT/'outputs/pure_dl_sw_ranked_probability_v155_fold5'
STAGES={'local':('SW',1024,True,1.,3e-4)}

def make_model(stage,master,metadata):return Parent(metadata,'local')

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'a100')
        self.register_buffer('rps_edges',evaluator.edges[1:-1].detach().clone())
        value=(math.log(DEFAULTS['SW'])-metadata['sw_target_median'])/metadata['sw_target_scale']
        self.register_buffer('rps_default_z',torch.tensor(value,dtype=torch.float32))
    def forward(self,output,labels,default,mask,quality):
        original=super().forward(output,labels,default,mask,quality)
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        y=labels[mask][:,2].float();d=default[mask][:,2];q=quality[mask].detach().float();parts=[]
        for begin in range(0,len(g),1024):
            take=slice(begin,begin+1024)
            parts.append(checkpoint(ranked_loss,g[take],r[take],y[take],d[take],self.rps_edges,self.rps_default_z,
                use_reentrant=False,preserve_rng_state=False))
        rps=(q*torch.cat(parts)).mean()/3
        return original[0]+rps,original[1],original[2]+rps

"""One joint raw encoder/state/density phase; matched coarse/fine normalization."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from sw_state_bound_v289 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective as ParentObjective,Network as Parent

STUDY=ROOT/'results/deep_learning/sw_coarse_fine_v299_20261002'
OUT=ROOT/'outputs/pure_dl_sw_coarse_fine_v299_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('flat','hierarchical')}

class GroupReadout(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.coarse=nn.Linear(48,8);self.fine=nn.Linear(48,256)
        self.modulation=nn.Parameter(torch.zeros(8,96))
        self.register_buffer('boundaries',torch.arange(0,257,32,dtype=torch.long))
        self.slices=[(32*g,32*(g+1)) for g in range(8)]
        nn.init.zeros_(self.coarse.weight);nn.init.constant_(self.coarse.bias,torch.tensor(32.).log().item())
        nn.init.zeros_(self.fine.weight);nn.init.zeros_(self.fine.bias)
    def fit_groups(self,counts):
        total=counts.sum();assert float(total)>0
        cumulative=counts.cumsum(0);bounds=[0]
        for k in range(1,8):
            proposed=int(torch.searchsorted(cumulative,total*k/8))+1
            bounds.append(max(bounds[-1]+1,min(proposed,256-(8-k))))
        bounds.append(256)
        with torch.no_grad():
            self.boundaries.copy_(torch.tensor(bounds,device=self.boundaries.device))
            self.coarse.bias.copy_(torch.tensor([b-a for a,b in zip(bounds[:-1],bounds[1:])],device=self.coarse.bias.device).float().log())
        self.sync_slices()
        return bounds
    def sync_slices(self):
        bounds=self.boundaries.detach().cpu().tolist()
        self.slices=list(zip(bounds[:-1],bounds[1:]))
    def forward(self,h):
        # Save/load uses persistent boundaries; Python slices are fixed train-fit metadata.
        coarse=self.coarse(h).float()
        if self.mode=='hierarchical':coarse=coarse.log_softmax(-1)
        pieces=[]
        for g,(lo,hi) in enumerate(self.slices):
            gamma,beta=self.modulation[g].chunk(2)
            feature=h*(1+.25*torch.tanh(gamma))+.25*torch.tanh(beta)
            fine=F.linear(feature,self.fine.weight[lo:hi],self.fine.bias[lo:hi]).float()
            if self.mode=='hierarchical':pieces.append(coarse[...,g,None]+fine.log_softmax(-1))
            else:pieces.append(coarse[...,g,None]+fine-fine.new_tensor(hi-lo).log())
        result=torch.cat(pieces,-1)
        return result if self.mode=='hierarchical' else result.log_softmax(-1)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'balanced');self.readout_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261299);self.readout=GroupReadout(mode)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        ix=data['train_raw'];keep=data['labeled'][ix]&(~data['default'][ix,2])&(data['quality'][ix]>0)
        ix=ix[keep];bins=torch.bucketize(data['labels'][ix,2].float().contiguous(),self.edges[1:-1])
        counts=torch.bincount(bins,weights=data['quality'][ix].float(),minlength=256)
        boundaries=self.readout.fit_groups(counts)
        info.update(coarse_fine_mode=self.readout_mode,training_Q_weighted_continuous_bin_groups=boundaries,groups=8,nonempty_groups_forced=True,group_fit_rows=len(ix),group_fit_counts_sha256=hashlib.sha256(counts.detach().cpu().numpy().tobytes()).hexdigest(),train_only_labels_for_group_fit=True,new_readout_zero_fine_uniform_initial=True,coarse_bias_log_group_size=True,nonzero_readout_initialization=False,group_conditioned_feature_modulation_bound=.25,new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),only_normalization_differs_in_pair=True,no289_or297_trained_weights_loaded=True,original_grid_actions_and_old104_prior_retained=True)
        return info
    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs);self.readout.sync_slices();return result

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'balanced')

def make_model(stage,master,metadata):return Network(master,metadata,stage)

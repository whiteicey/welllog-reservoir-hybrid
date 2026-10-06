"""Original154 prior plus directly supervised new raw-input categorical branch."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data
from perm_memory_readout_v268 import Evaluator,channel_objective as mixture_objective
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/perm_direct_multiscale_v271_20261001'
OUT=ROOT/'outputs/pure_dl_perm_direct_multiscale_v271_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('repeat','multiscale')}
class ScaleMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
    def forward(self,h):return h+self.net(h)
class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.base=Positive(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.base.requires_grad_(False).eval()
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][1],dtype=torch.float32));self.register_buffer('mix_alpha',torch.tensor(0.))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261271);self.stem=nn.Sequential(nn.Linear(207,48),nn.GELU());self.branches=nn.ModuleList([ScaleMLP() for _ in range(3)]);self.fuse=nn.Sequential(nn.Linear(144,48),nn.GELU());self.readout=nn.Linear(48,256)
        self.scales=(1,4,16) if mode=='multiscale' else (1,1,1);self.aux_logits=None
        self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.;self.last_aux_ce=0.
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.stem.parameters())+list(self.branches.parameters())+list(self.fuse.parameters()),lr=3e-4,name='PERM_new_multiscale_representation'),dict(params=list(self.readout.parameters()),lr=3e-4,name='PERM_direct_categorical_readout')]
    def set_epoch(self,epoch):self.mix_alpha.fill_(.1*min(epoch/5,1.))
    def initialize_priors(self,data):return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all154_weights_frozen=True,scales=list(self.scales),registered_capacity_same=True,nonzero_readout_initialization=True,auxiliary_direct_CE_weight=1.,fixed_alpha_linear_ramp_epochs=5,final_alpha=.1,initial_alpha=0.,direct_component_supervision_independent_of_alpha=True,no_old_repair_weights_loaded=True,input_features='original206+relative',one_added_PERM_stage=True,initial_prediction_prior_discretization_only=True,old_gate_frozen=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_logits=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability(1,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),dim=-1);validf=valid.float().unsqueeze(1);features=[]
        for scale,branch in zip(self.scales,self.branches):
            if scale==1:v=valid;z=inp
            else:
                den=F.avg_pool1d(validf,scale,scale);v=den.squeeze(1)>0
                num=F.avg_pool1d(inp.transpose(1,2).float()*validf,scale,scale);z=(num/den.clamp_min(1e-12)).transpose(1,2).to(inp.dtype)
            h=branch(self.stem(z)).masked_fill(~v.unsqueeze(-1),0)
            if scale!=1:h=F.interpolate(h.transpose(1,2),size=x.shape[1],mode='linear',align_corners=False).transpose(1,2)
            features.append(h.masked_fill(~valid.unsqueeze(-1),0))
        fused=self.fuse(torch.cat(features,dim=-1)).masked_fill(~valid.unsqueeze(-1),0)
        logits=self.readout(fused).float()
        if self.training:self.aux_logits=[self.readout(h).float() for h in features]+[logits]
        new=logits.softmax(-1);mass=(1-self.mix_alpha)*prior+self.mix_alpha*new
        mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    original=mixture_objective(output,labels,default,mask,quality,c,mode,model,well_ids)
    d=default[mask][:,1];z=labels[mask][:,1].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),model.edges[1:-1]);weights=q*(~d).float();den=weights.sum().clamp_min(1e-12)
    aux=sum((weights*F.cross_entropy(logits[mask].float(),bins,reduction='none')).sum()/den for logits in model.aux_logits)/len(model.aux_logits)
    model.last_aux_ce=float(aux.detach());model.aux_logits=None
    return original[0]+aux*q.mean()/3,original[1],original[2]+aux*q.mean()

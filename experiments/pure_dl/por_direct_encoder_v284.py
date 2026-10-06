"""Original154 prior plus directly supervised new raw-input categorical branch."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from por_ranked_probability_v151 import ROOT,SOURCE,Network as Positive,load_data as original_data
from sw_calibrated_state_v269 import Evaluator
from compact_por_mixed_family_v113 import probability
STUDY=ROOT/'results/deep_learning/por_direct_encoder_v284_20261002'
OUT=ROOT/'outputs/pure_dl_por_direct_encoder_v284_fold5'
PARENT=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt'
PARENT_SHA='39f2255b8fe61fa74607af027ece1dd352e5e2d0729b9bce95d1424c1b227603'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('raw',)}
class ScaleMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
    def forward(self,h):return h+self.net(h)
class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.metadata=metadata;self.base=Positive(master,'joint_rps')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.base.requires_grad_(False).eval()
        self.target_median=float(master['regression_median'][0]);self.target_scale=float(master['regression_scale'][0]);self.default_z=(__import__('math').log1p(.1)-self.target_median)/self.target_scale
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][0],dtype=torch.float32))
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][0],dtype=torch.float32));self.register_buffer('mix_alpha',torch.tensor(0.))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261271);self.stem=nn.Sequential(nn.Linear(207,48),nn.GELU());self.branches=nn.ModuleList([ScaleMLP() for _ in range(3)]);self.fuse=nn.Sequential(nn.Linear(144,48),nn.GELU());self.readout=nn.Linear(48,256)
        self.scales=(1,1,1);self.aux_features=None
        self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.;self.last_aux_ce=0.
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.stem.parameters())+list(self.branches.parameters())+list(self.fuse.parameters()),lr=3e-4,name='POR_new_raw_feature_backbone'),dict(params=list(self.readout.parameters()),lr=3e-4,name='POR_direct_categorical_readout')]
    def set_epoch(self,epoch):self.mix_alpha.fill_(.1*min(epoch/5,1.))
    def initialize_priors(self,data):
        ix=data['train_raw'];q=data['quality'][ix].detach().float();d=data['default'][ix,0].float();self.metadata['gate_default_prior']=float((q*d).sum()/q.sum())
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all151_prior_gate_frozen=True,new_raw_encoder_and_direct_readout_jointly_trainable=True,scales=[1,1,1],nonzero_readout_initialization=True,direct_aux_hard_soft_CE_weight=1.,schedule_ramp_epochs=5,final_mix_alpha=.1,final_tilt_gamma=1.,readout_tilt_bound=3.,initial_function_discretization_only=True,no_repair_weights_loaded=True,one_POR_additional_phase=True,new_active_parameters=57568,independent_supervision_already_existed_in262263=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_features=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability('t3_cauchy',0,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
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
        if self.training:self.aux_features=features+[fused]
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        if self.mode=='raw':mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)
        else:mass=(prior.log()+(self.mix_alpha/.1)*3*torch.tanh(logits/3)).softmax(-1)
        mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)


def load_data(stage):return original_data(stage)
def channel_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=None):
    assert target_index==0
    gate,logmass=output;g=gate[mask][:,0].float();lp=logmass[mask][:,0].float()
    d=default[mask][:,0];z=labels[mask][:,0].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),model.edges[1:-1]);truth=torch.expm1(z*model.target_scale+model.target_median)
    soft=(1-(model.physical_centers[None]-truth[:,None]).abs()/(.08*(truth.abs()[:,None]+1e-5))).clamp_min(0);den=soft.sum(-1,keepdim=True);soft=torch.where(den>0,soft/den.clamp_min(1e-12),F.one_hot(bins,256).float()).detach()
    hard=-lp.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*lp).sum(-1)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');likelihood=(q*(bce+ce.masked_fill(d,0))).mean()/3
    atom=(model.edges[1:-1]>=model.default_z).float()[None];joint=g.sigmoid()[:,None]*atom+(1-g.sigmoid()[:,None])*lp.exp().cumsum(-1)[:,:-1]
    target=torch.where(d,torch.full_like(z,model.default_z),z);target_cdf=(target[:,None]<=model.edges[None,1:-1]).float();rps=(q*(joint-target_cdf).square().mean(-1)).mean()/3
    weight=q*(~d).float();continuous=weight.sum().clamp_min(1e-12);terms=[]
    for h in model.aux_features:
        own=model.readout(h[mask].float()).log_softmax(-1);hard=-own.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*own).sum(-1);terms.append((weight*ce).sum()/continuous)
    aux=sum(terms)/len(terms);model.aux_features=None;model.last_aux_ce=float(aux.detach())
    return likelihood+rps+aux*q.mean()/3,likelihood,rps

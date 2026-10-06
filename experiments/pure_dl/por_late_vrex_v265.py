"""Frozen113 prior with copied late encoding and retained151 readout joint update."""
import copy,hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from por_ranked_probability_v151 import ROOT,SOURCE,Network as Parent,Evaluator,load_data,channel_objective as original_loss
from compact_por_mixed_family_v113 import Network as Base113,choose
from compact_por_cauchy_v112 import IQR_SCALE
STUDY=ROOT/'results/deep_learning/por_late_vrex_v265_20261001'
OUT=ROOT/'outputs/pure_dl_por_late_vrex_v265_fold5'
PARENT=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt'
PARENT_SHA='39f2255b8fe61fa74607af027ece1dd352e5e2d0729b9bce95d1424c1b227603'
STAGES={a:('POR',256,False,0.,3e-5) for a in ('control','vrex')}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,'joint_rps');self.mode=mode
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.requires_grad_(False)
        self.late_dec0=nn.ModuleList([copy.deepcopy(m.dec0) for m in self.backbone.members]);self.late_combine=nn.ModuleList([copy.deepcopy(m.combine) for m in self.backbone.members])
        self.late_dec0.requires_grad_(True);self.late_combine.requires_grad_(True);self.context_head.requires_grad_(True)
        self.late_inputs={};self.late_points={};self.auxiliary_ramp=0.;self.last_vrex=0.;self.last_groups=0
        self.extra_hooks=[]
        for i,m in enumerate(self.backbone.members):
            self.extra_hooks.append(m.dec0.register_forward_pre_hook(self.capture_late(i)))
            self.extra_hooks.append(m.combine.register_forward_pre_hook(self.capture_point(i)))
    def capture_late(self,i):
        def hook(module,args):self.late_inputs[i]=args[0].detach()
        return hook
    def capture_point(self,i):
        def hook(module,args):self.late_points[i]=args[0][:,64:].detach()
        return hook
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.late_dec0.eval();self.late_combine.eval();return self
    def groups(self):return [dict(params=list(self.late_dec0.parameters())+list(self.late_combine.parameters()),lr=1e-5,name='POR_copied_existing_late_encoder'),dict(params=list(self.context_head.parameters()),lr=3e-5,name='POR_retained151_readout')]
    def initialize_priors(self,data):return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,original113_prior_and_gate_frozen=True,retained151_readout_updated=True,new_copied_late_encoder=True,new_inference_parameters=sum(p.numel() for p in self.late_dec0.parameters())+sum(p.numel() for p in self.late_combine.parameters()),initial_function_expected151=True,late_dropout_eval=True,one_added_POR_stage=True,no_trained_repair_weights=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.late_inputs.clear();self.late_points.clear();self.local_captured.clear()
        with torch.no_grad():gate,raw=Base113.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        hidden=[c(torch.cat((b(self.late_inputs[i]),self.late_points[i]),1)) for i,(b,c) in enumerate(zip(self.late_dec0,self.late_combine))]
        features=torch.cat((*hidden,x[...,206:].transpose(1,2)),1)
        u=self.context_head(features,valid);ref=F.softplus(raw[:,:,0,2].detach())+.025
        delta=ref*torch.tanh(u[...,:16]/ref);weight_delta=torch.tanh(u[...,16:]);own=raw[:,:,0]
        self.late_inputs.clear();self.late_points.clear();self.local_captured.clear()
        return gate,torch.stack((own[:,:,0]+weight_delta,own[:,:,1]+delta,own[:,:,2]),2).unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,stage)
def continuous_bin_ce(raw,z,edges):
    r=raw.float();sd=F.softplus(r[:,2])+.025;delta=edges[None,:,None]-r[:,1,None,:];phi=torch.atan(delta/sd[:,None,:])
    cdf=choose('t3_cauchy',.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi,.5+torch.atan(delta/(sd[:,None,:]*IQR_SCALE))/math.pi)
    cdf=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1);cdf=torch.cat((cdf.new_zeros(len(r),1),cdf,cdf.new_ones(len(r),1)),-1)
    mass=cdf.diff(-1).clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True);bins=torch.bucketize(z.contiguous(),edges)
    return -mass.gather(1,bins[:,None]).squeeze(1).log()
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    original=original_loss(output,labels,default,mask,quality,0,'joint_rps',model)
    raw=output[1][mask][:,0].float();z=labels[mask][:,0].float();d=default[mask][:,0];q=quality[mask].detach().float();ids=well_ids[mask]
    use=~d;r=raw[use];y=z[use];w=q[use];ids=ids[use]
    if len(r):
        parts=[checkpoint(continuous_bin_ce,a,b,model.rps_edges,use_reentrant=False,preserve_rng_state=False) for a,b in zip(r.split(512),y.split(512))];ce=torch.cat(parts)
        groups,inv=torch.unique(ids,sorted=True,return_inverse=True);den=torch.zeros(len(groups),device=w.device).scatter_add(0,inv,w);counts=torch.zeros_like(den).scatter_add(0,inv,torch.ones_like(w));num=torch.zeros_like(den).scatter_add(0,inv,w*ce)
        means=num[counts>=8]/den[counts>=8].clamp_min(1e-12)
        penalty=means.var(unbiased=False)/means.detach().square().mean().clamp_min(1e-4) if len(means)>=2 else ce.sum()*0
        model.last_vrex=float(penalty.detach());model.last_groups=len(means)
    else:penalty=raw.sum()*0;model.last_vrex=0.;model.last_groups=0
    beta=.1*model.auxiliary_ramp if mode=='vrex' else 0.
    return original[0]+beta*penalty/3,original[1],original[2]

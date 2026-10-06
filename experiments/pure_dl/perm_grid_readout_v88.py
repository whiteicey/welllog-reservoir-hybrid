"""Frozen private PERM predictor plus bounded conditional probability-grid readout."""
from pathlib import Path
import copy,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_split_route_v29 import Network as FrozenLoader
from champion_all_heads_v26 import load_data,TARGETS
from train_scaling_v21 import Evaluator as BaseEvaluator
from train_dual8222_heads80 import SOURCE
from train_invariant import conditional_probability
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/perm_grid_readout_v88_20260930'
PREVIOUS=ROOT/'results/deep_learning/perm_full_utility_v87_20260930'
MODES=('likelihood','joint')
LEARNING_RATE=1e-4

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class GridHead(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv1d(319,64,1);self.output=nn.Conv1d(64,257,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,features,valid):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            proposal=self.output(F.gelu(self.input(features))).transpose(1,2).float()
        return proposal.masked_fill(~valid[...,None],0)
    def transform(self,proposal,mask,gate,prior):
        delta=proposal[mask];g=gate+.25*torch.tanh(delta[:,0]/.25)
        log_ratio=.5*torch.tanh(delta[:,1:]/.5);mass=prior*log_ratio.exp()
        # At delta=0 the denominator is exactly1; preserve the original mass
        # arithmetic and therefore the old argmax function at initialization.
        ratio=mass.sum(-1,keepdim=True)/prior.sum(-1,keepdim=True).clamp_min(1e-12)
        p=mass/ratio.clamp_min(1e-12)
        return g,p

class Network(nn.Module):
    def __init__(self,target,master,modes=MODES):
        super().__init__();assert target=='PERM';self.target=target
        self.frozen=FrozenLoader(target,master,modes=('coupled',))
        source=ROOT/'outputs/pure_dl_split_route_v29_PERM_fold5/coupled/model_best.pt'
        cp=torch.load(source,map_location='cpu',weights_only=False)
        self.frozen.heads['coupled'].load_state_dict(cp['delta_state'],strict=True)
        self.frozen.requires_grad_(False);self.frozen.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261188);initial=GridHead()
        self.heads=nn.ModuleDict({m:copy.deepcopy(initial) for m in modes})
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][1],dtype=torch.float32))
        self.metadata=dict(source_checkpoint=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_epoch=cp['best_epoch'],source_score=cp['target_score'],added_parameters_per_candidate=37185,
            old_predictor_frozen=True,only_own_private_hidden_and_existing_side_inputs=True,
            log_mass_bound=.5,gate_logit_bound=.25,initial_output_zero=True,
            prior_total_mass_preserved=True,prior_zero_bins_cannot_gain_mass=True,
            density_expression='bounded multiplicative residual over original256probabilitygrid',
            target_supervision='own training log10 target, two adjacent support centers',
            utility_coefficient={'likelihood':0.,'joint':1.},temperature=.05)
        self.train(False)
        assert all(sum(p.numel() for p in h.parameters())==37185 for h in self.heads.values())
    def train(self,mode=True):
        nn.Module.train(self,False);self.training=mode;self.heads.train(mode);return self
    def frozen_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('heads.')}
    def trainable_groups(self,mode):return [dict(params=self.heads[mode].parameters(),lr=1e-4,name='grid_head')]
    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        self.frozen.eval();h,o,ctx,_=self.frozen.extract(x,relative,valid,absolute)
        output=self.frozen.heads['coupled'](h,o,ctx,x[...,206:],valid)
        features=torch.cat((h[0],h[1],x[...,206:].transpose(1,2).to(h[0].dtype)),1)
        return features.masked_fill(~valid[:,None],0).detach(),output
    @torch.no_grad()
    def prior(self,output,mask):
        go,raw=output;g=go[mask][:,0].float();r=raw[mask][:,0].float()
        p=conditional_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        return g,p
    def close(self):self.frozen.close()

class Evaluator(BaseEvaluator):
    @torch.no_grad()
    def many(self,model,modes):
        model.eval();d=self.data;modes=tuple(modes)
        predictions={m:np.full(len(d['validation']),np.nan,np.float64) for m in modes};coverage=np.zeros(len(d['validation']),np.int8)
        from multiscale_sequence import window_indices
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            features,old=model.extract(d['x'][ix],rel,valid,d['absolute'][ix])
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()];use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;mask=torch.zeros_like(valid);mask[:,64:192]=torch.as_tensor(use,device='cuda')
            gate,prior=model.prior(old,mask)
            for m in modes:
                proposal=model.heads[m](features,valid);g,p=model.heads[m].transform(proposal,mask,gate,prior)
                pg=g.sigmoid();mass=torch.cat((pg[:,None],(1-pg[:,None])*p),-1)
                predictions[m][loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all();result={}
        for m,pred in predictions.items():
            assert np.isfinite(pred).all();combined=self.reference.copy();combined[:,1]=pred
            scores=base.metric(d['truth'][d['validation']],combined)
            result[m]=(pred,float(scores['PERM']),float(scores['TOTAL']))
        return result

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,master,mode):
        super().__init__();self.alpha=0. if mode=='likelihood' else 1.
        centers=torch.as_tensor(master['centers'][1],device='cuda',dtype=torch.float32)
        self.median=float(master['regression_median'][1]);self.scale=float(master['regression_scale'][1])
        self.register_buffer('z_centers',(centers.log10()-self.median)/self.scale)
        self.register_buffer('log_actions',torch.as_tensor(evaluator.actions,device='cuda',dtype=torch.float32).clamp_min(1e-5).log10())
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
    def forward(self,output,labels,default,mask,quality):
        g,p=output;z=labels[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float()
        right=torch.searchsorted(self.z_centers,z.contiguous()).clamp(1,255);left=right-1
        t=((z-self.z_centers[left])/(self.z_centers[right]-self.z_centers[left])).clamp(0,1)
        logp=p.clamp_min(1e-12).log();a=logp.gather(1,left[:,None])[:,0];b=logp.gather(1,right[:,None])[:,0]
        ce=(-(1-t)*a-t*b).masked_fill(d,0);bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        likelihood=(q*(bce+ce)).mean()/3
        pg=g.sigmoid();mass=torch.cat((pg[:,None],(1-pg[:,None])*p),-1)
        ap=((mass@self.utilities)/.05).softmax(-1)
        log_y=torch.where(d,torch.full_like(z,-2.),z*self.scale+self.median)
        utility=(1-(self.log_actions[None]-log_y[:,None]).abs()).clamp_min(0)
        reward=(ap*utility).sum(-1);decision=(q*(1-reward)).mean()/3
        return likelihood+self.alpha*decision,likelihood,decision

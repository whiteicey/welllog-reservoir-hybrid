"""UNADOPTED CPU-tested prototype; not anomaly detection and not enabled in training.
High loss plus no progress identifies a training behavior, not corrupt labels.
Any future weighting requires a separate decision and independent quality evidence.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def mixture_nll(raw,target):
    r=raw.float();y=target.float();lw=r[:,:,0].log_softmax(-1);mu=r[:,:,1];sd=F.softplus(r[:,:,2])+.025
    return -(lw-.5*((y[...,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)

def _weighted_head(output,target,default,mask,weights):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        n=target.shape[1]//gate.shape[1];assert n*gate.shape[1]==target.shape[1]
        gate=gate.repeat_interleave(n,1);raw=raw.repeat_interleave(n,1)
    assert bool(mask.any())
    d=default[mask].bool();active=~d;gate=gate[mask].float()
    nll=mixture_nll(raw[mask],target[mask].masked_fill(d,0)).masked_fill(d,0)
    if weights is None:w=torch.ones_like(nll)
    else:
        assert weights.shape==target.shape
        w=weights[mask].detach().float();assert bool(torch.isfinite(w).all()) and bool((w>0).all())
    count=active.sum(0);average=(w*active).sum(0)/count.clamp_min(1)
    normalized=w/average.clamp_min(1e-8)
    continuous=(nll*normalized).masked_fill(d,0).mean()
    return F.binary_cross_entropy_with_logits(gate,d.float())+continuous

def weighted_objective(outputs,target,default,mask,deep_supervision=False,weights=None):
    if not deep_supervision:return _weighted_head(outputs,target,default,mask,weights)
    main,mid,coarse=outputs
    return _weighted_head(main,target,default,mask,weights)+.2*_weighted_head(mid,target,default,mask,weights)+.1*_weighted_head(coarse,target,default,mask,weights)

class BlockWeightTracker:
    """Prototype status only. Count repeated observed rows, not unique examples."""
    def __init__(self,block_ids_gpu,train_raw_gpu,num_blocks,out_dir):
        self.ids=block_ids_gpu.long().detach();self.device=self.ids.device;self.num_blocks=int(num_blocks);self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=True)
        assert self.ids.ndim==1 and bool(((self.ids>=0)&(self.ids<num_blocks)).all())
        self.train_rows=torch.zeros(len(self.ids),dtype=torch.bool,device=self.device);self.train_rows[train_raw_gpu]=True
        self.train_blocks=torch.zeros(num_blocks,dtype=torch.bool,device=self.device);self.train_blocks[self.ids[train_raw_gpu]]=True
        shape=(num_blocks,3);self.cached=torch.ones(shape,device=self.device);self.ema=torch.full(shape,float('nan'),device=self.device);self.streak=torch.zeros(shape,dtype=torch.long,device=self.device)
        self.sums=torch.zeros(shape,device=self.device,dtype=torch.float64);self.counts=torch.zeros(shape,device=self.device,dtype=torch.long);self.open=False;self.last_epoch=0
    @torch.no_grad()
    def begin_epoch(self):
        assert not self.open;self.sums.zero_();self.counts.zero_();self.open=True
    @torch.no_grad()
    def weights(self,raw_ix):return self.cached[self.ids[raw_ix]].detach()
    @torch.no_grad()
    def observe(self,raw_main,labels,defaults,supervised,raw_ix):
        assert self.open and raw_ix.shape==supervised.shape and labels.shape==defaults.shape
        ix=raw_ix[supervised];assert bool(self.train_rows[ix].all()),'Validation/non-training row passed as supervised'
        if not ix.numel():return
        d=defaults[supervised].bool();active=~d;target=labels[supervised].masked_fill(d,0);nll=mixture_nll(raw_main[supervised].detach(),target)
        assert bool(torch.isfinite(nll[active]).all())
        blocks=self.ids[ix]
        for j in range(3):
            use=active[:,j];b=blocks[use]
            self.sums[:,j].scatter_add_(0,b,nll[use,j].double());self.counts[:,j].scatter_add_(0,b,torch.ones_like(b))
    @torch.no_grad()
    def end_epoch(self,epoch):
        assert self.open and epoch>self.last_epoch;self.open=False;self.last_epoch=int(epoch)
        valid=(self.counts>=64)&self.train_blocks[:,None]
        old=self.ema.clone();mean=(self.sums/self.counts.clamp_min(1)).float()
        new=torch.where(torch.isfinite(old),.8*old+.2*mean,mean)
        self.ema=torch.where(valid,new,old)
        self.cached.fill_(1);self.streak[~valid]=0
        thresholds=[];high_counts=[]
        for j in range(3):
            use=valid[:,j]
            if bool(use.any()):
                threshold=torch.quantile(self.ema[use,j],.9);thresholds.append(float(threshold))
                high=use&(self.ema[:,j]>=threshold);no_progress=torch.isfinite(old[:,j])&(self.ema[:,j]>=old[:,j]);flag=high&no_progress
                self.streak[:,j]=torch.where(flag,self.streak[:,j]+1,torch.zeros_like(self.streak[:,j]));high_counts.append(int(high.sum()))
                if epoch>=10:self.cached[:,j]=torch.where(use&(self.streak[:,j]>=3),.5,1.)
            else:thresholds.append(None);high_counts.append(0)
        assert bool((self.cached[~self.train_blocks]==1).all())
        result={'prototype_status':'unadopted; not anomaly detection','epoch':int(epoch),'applies_to_next_epoch':int(epoch)+1,'ema_alpha':.2,'minimum_observations':64,'counts_are_repeated_observations_not_unique_rows':True,'threshold_p90_ema':thresholds,'eligible_block_channels':valid.sum(0).cpu().tolist(),'sparse_training_block_channels':((~valid)&self.train_blocks[:,None]).sum(0).cpu().tolist(),'high_ema_block_channels':high_counts,'downweighted_block_channels':(self.cached<1).sum(0).cpu().tolist(),'raw_weight_values':[.5,1.],'normalized_loss_weight_range_may_be':[.5,2.]}
        (self.out/f'epoch_{epoch:03d}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    def finalize(self):
        assert not self.open
        path=self.out/'dynamic_block_weight_final.npz'
        np.savez_compressed(path,weights=self.cached.cpu().numpy(),ema=self.ema.cpu().numpy(),streak=self.streak.cpu().numpy(),train_block_mask=self.train_blocks.cpu().numpy(),last_epoch=np.asarray(self.last_epoch))
        return {'path':str(path),'last_epoch':self.last_epoch,'prototype_status':'unadopted; not anomaly detection'}

def self_test():
    torch.set_num_threads(2);torch.manual_seed(851)
    from population_likelihood_sequence import objective as reference
    raw=torch.randn(2,8,3,3,8,requires_grad=True);gate=torch.randn(2,8,3,requires_grad=True);y=torch.randn(2,8,3);d=torch.rand(2,8,3)>.4;mask=torch.ones(2,8,dtype=torch.bool);mask[:,0]=False
    w=torch.ones_like(y,requires_grad=True)
    loss=weighted_objective((gate,raw),y,d,mask,weights=w);expected=reference((gate,raw),y,d,mask);assert torch.equal(loss,expected)
    rr=raw[mask];dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    assert torch.allclose(mixture_nll(rr,y[mask]),-dist.log_prob(y[mask]),atol=1e-6)
    weights=torch.where(torch.rand_like(y)>.5,.5,1.).requires_grad_();actual=weighted_objective((gate,raw),y,d,mask,weights=weights)
    active=~d[mask];ww=weights[mask].detach();normalizer=torch.stack([ww[active[:,j],j].mean() if active[:,j].any() else torch.tensor(1.) for j in range(3)])
    independent=F.binary_cross_entropy_with_logits(gate[mask],d[mask].float())+(-dist.log_prob(y[mask])*ww/normalizer).masked_fill(~active,0).mean();assert torch.allclose(actual,independent,atol=1e-6)
    actual.backward();assert weights.grad is None and bool((raw.grad[d]==0).all()) and bool((raw.grad[~mask]==0).all())
    all_default=torch.ones_like(d);assert torch.equal(weighted_objective((gate,raw),y,all_default,mask,weights=weights),F.binary_cross_entropy_with_logits(gate[mask],all_default[mask].float()))
    altered=y.clone();altered[d|~mask[:,:,None]]=float('nan');assert torch.equal(actual,weighted_objective((gate,raw),altered,d,mask,weights=weights))
    outputs=((gate,raw),(gate[:,::2],raw[:,::2]),(gate[:,::4],raw[:,::4]));assert torch.equal(weighted_objective(outputs,y,d,mask,True,w),reference(outputs,y,d,mask,True))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='welllog-blockweight-test-') as directory:
        ids=torch.arange(192)//64;tracker=BlockWeightTracker(ids,torch.arange(128),3,directory)
        ix=torch.arange(128).reshape(1,-1);r=torch.zeros(1,128,3,3,8);labels=torch.zeros(1,128,3);labels[:,64:,0]=4.;defaults=torch.zeros_like(labels,dtype=torch.bool);defaults[:,:,1]=True;defaults[:,::2,2]=True;supervised=torch.ones_like(ix,dtype=torch.bool)
        for epoch in range(1,11):tracker.begin_epoch();tracker.observe(r,labels,defaults,supervised,ix);tracker.end_epoch(epoch)
        assert tracker.cached[1,0]==.5 and tracker.cached[0,0]==1 and bool((tracker.cached[:,1:]==1).all());assert bool(torch.isnan(tracker.ema[2]).all());assert not tracker.weights(ix).requires_grad
        labels[:,64:,0]=0;tracker.begin_epoch();tracker.observe(r,labels,defaults,supervised,ix);tracker.end_epoch(11);assert tracker.cached[1,0]==1 and tracker.streak[1,0]==0
        tracker.begin_epoch()
        try:tracker.observe(r[:,:1],labels[:,:1],defaults[:,:1],supervised[:,:1],torch.tensor([[128]]))
        except AssertionError:pass
        else:raise AssertionError('Validation observation accepted')
        tracker.end_epoch(12);assert bool((tracker.cached==1).all()) and bool((tracker.streak==0).all());tracker.finalize()
    return {'passed':True,'unadopted_prototype_only':True,'independent_mixture_nll':True,'unit_weights_exact_population_loss':True,'weighted_formula_and_auxiliary':True,'default_and_unsupervised_zero_gradients':True,'detached_weights':True,'sparse_reset_and_progress_recovery':True,'validation_update_rejected':True,'no_anomaly_claim':True}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))

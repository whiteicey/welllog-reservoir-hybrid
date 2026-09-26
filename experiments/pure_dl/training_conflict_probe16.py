"""Read-only fixed training-well probes. No weighting, resampling, or validation access."""
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from multiscale_sequence import window_indices
from train_invariant import conditional_probability

def point_statistics(gate, raw, target, default, density_edges):
    raw=raw.float();gate=gate.float();target=target.float();default=default.bool()
    lw=raw[:,:,0].log_softmax(-1);w=lw.exp();mu=raw[:,:,1];sd=torch.nn.functional.softplus(raw[:,:,2])+.025
    nll=-(lw-.5*((target[...,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    nll=nll.masked_fill(default,float('nan'))
    probabilities=[];entropies=[]
    for j in range(3):
        cp=conditional_probability(w[:,j],mu[:,j],sd[:,j],torch.as_tensor(density_edges[j],device=raw.device,dtype=torch.float32))
        probabilities.append(cp);entropies.append(-(cp*cp.clamp_min(1e-30).log()).sum(-1)/math.log(cp.shape[-1]))
    cp=torch.stack(probabilities,1);pg=gate.sigmoid()
    target_cdf=(w*torch.special.ndtr((target[...,None]-mu)/sd)).sum(-1).clamp(0,1).masked_fill(default,float('nan'))
    target_tail=2*torch.minimum(target_cdf,1-target_cdf)
    ge=-(pg*pg.clamp_min(1e-30).log()+(1-pg)*(1-pg).clamp_min(1e-30).log())/math.log(2)
    return {'target_cdf':target_cdf,'target_tail':target_tail,'nll':nll,'conditional_entropy':torch.stack(entropies,1),'gate_probability':pg,'gate_entropy':ge,'conditional_mean_z':(w*mu).sum(-1),'gate_prediction':pg>=.5},cp.cumsum(-1)

def conditional_mean_utility(mean_z,true_z,median,scale):
    pred=np.asarray(mean_z,dtype=np.float64)*scale+median
    truth=np.asarray(true_z,dtype=np.float64)*scale+median
    values=[]
    for j in range(3):
        if j==1:u=1-np.abs(np.maximum(pred[:,j],-5)-np.maximum(truth[:,j],-5))
        else:
            inv=np.expm1 if j==0 else np.exp
            p=inv(np.clip(pred[:,j],-80,80));y=inv(np.clip(truth[:,j],-80,80))
            u=1-np.abs(p-y)/((.08 if j==0 else .05)*(np.abs(y)+1e-5))
        values.append(np.maximum(0,u))
    return np.stack(values,1)

class TrainingDynamicsProbe:
    def __init__(self,out_dir,train_wells,starts,labeled,lower,upper,depth,absolute,labels,defaults,density_edges,regression_median,regression_scale,target_names):
        self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=True)
        self.device=depth.device;self.lower=lower;self.upper=upper;self.depth=depth;self.absolute=absolute;self.labels=labels;self.defaults=defaults;self.labeled=labeled
        self.edges=density_edges;self.median=np.asarray(regression_median);self.scale=np.asarray(regression_scale);self.names=list(target_names)
        self.starts=np.asarray(starts);self.train_wells=set(map(int,train_wells));assert len(self.train_wells)==64
        centers=[];anchors=[];self.window_wells=[]
        for wi in sorted(self.train_wells):
            start,end=int(starts[wi]),int(starts[wi+1]);grid=np.arange(start,end,128)+64
            chosen=np.unique(np.linspace(0,len(grid)-1,min(16,len(grid)),dtype=int))
            for center in grid[chosen]:centers.append(int(center));anchors.append(min(int(center),end-1));self.window_wells.append(wi)
        self.centers=torch.tensor(centers,device=self.device);self.anchors=torch.tensor(anchors,device=self.device)
        raw_ids=[];well_ids=[]
        with torch.no_grad():
            for begin in range(0,len(centers),16):
                ix,exists=window_indices(self.centers[begin:begin+16],lower,upper,anchors=self.anchors[begin:begin+16])
                core=ix[:,64:192];use=exists[:,64:192]&labeled[core]
                for row,wi in enumerate(self.window_wells[begin:begin+16]):
                    ids=core[row][use[row]].cpu().numpy();raw_ids.extend(ids.tolist());well_ids.extend([wi]*len(ids))
        self.raw=np.asarray(raw_ids,np.int64);self.well=np.asarray(well_ids,np.int32)
        assert len(self.raw)>0 and len(np.unique(self.raw))==len(self.raw)
        self.block=((self.raw-self.starts[self.well])//128).astype(np.int32)
        ids=torch.as_tensor(self.raw,device=self.device);self.default=defaults[ids].detach().cpu().numpy().astype(bool);self.truth_z=labels[ids].detach().cpu().numpy()
        assert bool(labeled[ids].all()) and set(self.well).issubset(self.train_wells)
        self.epochs=[];self.history={};self.previous_cdf=None;self.probe_seconds=0.;self.probe_peak_gpu_mib=0.
    def record(self,model,x,epoch):
        started=time.perf_counter()
        assert not self.epochs or epoch>self.epochs[-1]
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None
        modes=[(m,m.training) for m in model.modules()];chunks=[];cdfs=[];seen=[]
        try:
            model.eval()
            with torch.no_grad():
                for begin in range(0,len(self.centers),16):
                    anchor=self.anchors[begin:begin+16]
                    ix,exists=window_indices(self.centers[begin:begin+16],self.lower,self.upper,anchors=anchor)
                    relative=torch.tanh((self.depth[ix]-self.depth[anchor,None])/10)
                    with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                        gate,raw=model(x[ix],relative,exists,absolute_depth=self.absolute[ix])
                    core=ix[:,64:192];use=exists[:,64:192]&self.labeled[core];ids=core[use]
                    stats,cdf=point_statistics(gate[:,64:192][use],raw[:,64:192][use],self.labels[ids],self.defaults[ids],self.edges)
                    chunks.append({k:v.cpu().numpy() for k,v in stats.items()});cdfs.append(cdf.cpu().numpy());seen.extend(ids.cpu().tolist())
        finally:
            for m,training in modes:m.training=training
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            assert torch.equal(cpu_rng,torch.get_rng_state())
            if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
            assert all(m.training==training for m,training in modes)
        assert np.array_equal(self.raw,np.asarray(seen))
        current={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};cdf=np.concatenate(cdfs).astype(np.float32)
        current['cdf_l1_change']=np.full_like(current['nll'],np.nan) if self.previous_cdf is None else np.abs(cdf-self.previous_cdf).mean(-1)
        self.previous_cdf=cdf
        current['hard_p90']=np.zeros_like(self.default)
        thresholds=[]
        for j in range(3):
            active=~self.default[:,j];threshold=float(np.quantile(current['nll'][active,j],.9)) if active.any() else float('nan');thresholds.append(threshold)
            current['hard_p90'][:,j]=active&(current['nll'][:,j]>=threshold)
        utility=conditional_mean_utility(current['conditional_mean_z'],self.truth_z,self.median,self.scale)
        current['mean_probe_utility']=np.where(self.default,np.nan,utility).astype(np.float32)
        current['mean_probe_correct']=(utility>=.5)&~self.default
        self.epochs.append(int(epoch))
        for k,v in current.items():self.history.setdefault(k,[]).append(v.astype(bool if v.dtype==bool else np.float32))
        self.history.setdefault('p90_threshold',[]).append(np.asarray(thresholds,np.float32))
        seconds=time.perf_counter()-started;self.probe_seconds+=seconds
        if self.device.type=='cuda':self.probe_peak_gpu_mib=max(self.probe_peak_gpu_mib,torch.cuda.max_memory_allocated(self.device)/1024**2)
        result={'epoch':int(epoch),'points':len(self.raw),'p90_threshold':thresholds,'training_wells_only':True,'seconds':seconds,'total_probe_seconds':self.probe_seconds,'cumulative_gpu_peak_mib_no_reset':self.probe_peak_gpu_mib}
        print('PROBE_DONE',json.dumps(result),flush=True)
        (self.out/'training_dynamics_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    def finalize(self):
        if not self.epochs:return {'records':0}
        np.savez_compressed(self.out/'training_dynamics_history.npz',epochs=np.asarray(self.epochs),raw_ids=self.raw,well_ids=self.well,block_ids=self.block,default=self.default,**{k:np.stack(v) for k,v in self.history.items()})
        h={k:np.stack(v) for k,v in self.history.items()};active=~self.default
        forget=(h['mean_probe_correct'][:-1]&~h['mean_probe_correct'][1:]).sum(0)
        def avg(a):
            finite=np.asarray(a)[np.isfinite(a)];return float(finite.mean()) if finite.size else None
        def summary(select,j):
            use=select&active[:,j]
            return {'target':self.names[j],'nondefault_points':int(use.sum()),'records':len(self.epochs),'hard_frequency_mean':avg(h['hard_p90'][:,use,j]),'persistent_hard_fraction_ge_half':avg(h['hard_p90'][:,use,j].mean(0)>=.5) if use.any() else None,'nll_initial':avg(h['nll'][0,use,j]),'nll_final':avg(h['nll'][-1,use,j]),'nll_drop':avg(h['nll'][0,use,j]-h['nll'][-1,use,j]),'conditional_entropy_mean':avg(h['conditional_entropy'][:,use,j]),'cdf_l1_change_mean':avg(h['cdf_l1_change'][:,use,j]),'forgetting_transitions':int(forget[use,j].sum()),'forgetting_per_point':avg(forget[use,j]),'gate_entropy_mean_all':avg(h['gate_entropy'][:,select,j]),'gate_error_mean_all':avg(h['gate_prediction'][:,select,j]!=self.default[select,j])}
        channel=[summary(np.ones(len(self.raw),bool),j) for j in range(3)]
        rows=[]
        for wi in sorted(set(self.well)):
            for j in range(3):rows.append(dict(well=int(wi),block=-1,scope='well',**summary(self.well==wi,j)))
            for block in sorted(set(self.block[self.well==wi])):
                for j in range(3):rows.append(dict(well=int(wi),block=int(block),scope='block',**summary((self.well==wi)&(self.block==block),j)))
        with (self.out/'training_dynamics_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result={'total_probe_seconds':self.probe_seconds,'cumulative_gpu_peak_mib_no_reset':self.probe_peak_gpu_mib,'gpu_peak_scope':'Observed cumulative process peak, includes training; probe does not reset allocator statistics','records':len(self.epochs),'epochs':self.epochs,'points':len(self.raw),'train_wells':sorted(self.train_wells),'channels':channel,'scope':'Fixed training probes only; observational, no weighting. Conditional Gaussian-mixture mean probe utility is not final utility-decoded prediction. Forgetting counts only >=0.5 to <0.5 transitions between recorded epochs on nondefault rows; unobserved epochs unknown. p90 computed separately each record and channel on nondefault probe points, not all training rows. CDF L1 is mean absolute change across 256 CDF entries, not Wasserstein distance. Sampling is sixteen deterministic inference-grid windows per training well, not all training rows.'}
        (self.out/'training_dynamics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result

def self_test():
    torch.manual_seed(71);raw=torch.randn(20,3,3,8,requires_grad=True);gate=torch.randn(20,3);y=torch.randn(20,3);d=torch.rand(20,3)>.6;edges=[torch.linspace(-5,5,257) for _ in range(3)];edges=[torch.cat([torch.tensor([-float('inf')]),e[1:-1],torch.tensor([float('inf')])]) for e in edges]
    stats,cdf=point_statistics(gate,raw,y,d,edges)
    distribution=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=raw[:,:,0]),torch.distributions.Normal(raw[:,:,1],torch.nn.functional.softplus(raw[:,:,2])+.025))
    from scipy.special import ndtr
    weights=raw[:,:,0].detach().softmax(-1).numpy();means=raw[:,:,1].detach().numpy();stds=(torch.nn.functional.softplus(raw[:,:,2])+.025).detach().numpy()
    independent_cdf=(weights*ndtr((y.numpy()[...,None]-means)/stds)).sum(-1)
    assert np.max(abs(independent_cdf[~d.numpy()]-stats['target_cdf'].detach().numpy()[~d.numpy()]))<2e-7
    assert torch.isnan(stats['target_cdf'][d]).all()
    extreme=y.clone();extreme[:,0]=-float('inf');extreme[:,1]=float('inf')
    extreme_stats,_=point_statistics(gate,raw,extreme,torch.zeros_like(d),edges)
    assert bool((extreme_stats['target_cdf'][:,0]==0).all()) and torch.allclose(extreme_stats['target_cdf'][:,1],torch.ones(20),atol=2e-7)
    expected=-distribution.log_prob(y);assert torch.allclose(stats['nll'][~d],expected[~d],atol=1e-6);assert torch.isnan(stats['nll'][d]).all()
    npcp=np.diff(np.concatenate([np.zeros((20,3,1)),cdf.detach().numpy()],-1),axis=-1);entropy=-np.sum(npcp*np.log(np.maximum(npcp,1e-30)),-1)/np.log(256)
    assert np.max(abs(entropy-stats['conditional_entropy'].detach().numpy()))<2e-6
    stats['nll'].nan_to_num().sum().backward();assert torch.equal(raw.grad[d],torch.zeros_like(raw.grad[d]))
    assert torch.equal(stats['gate_prediction'],gate>=0)
    means=np.zeros((2,3));u=conditional_mean_utility(means,means,np.array([1.,0.,3.]),np.ones(3));assert np.all(u==1)
    correct=np.array([[True,False],[False,True],[True,False]]);assert np.array_equal((correct[:-1]&~correct[1:]).sum(0),[1,1])
    # Small CPU integration checks actual row selection, RNG/mixed modes and persistence.
    import tempfile
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.child=torch.nn.Dropout();self.child.eval()
        def forward(self,x,relative,exists,absolute_depth=None):
            return torch.randn(*x.shape[:2],3),torch.randn(*x.shape[:2],3,3,8)
    starts=np.arange(65)*32;n=int(starts[-1]);well=np.repeat(np.arange(64),32)
    lower=torch.tensor(starts[well]);upper=lower+32;depth=torch.arange(n,dtype=torch.float32);labels=torch.randn(n,3);defaults=torch.rand(n,3)>.5;labeled=torch.ones(n,dtype=torch.bool);labeled[::7]=False
    with tempfile.TemporaryDirectory(prefix='welllog-probe-selftest-') as directory:
        probe=TrainingDynamicsProbe(directory,np.arange(64),starts,labeled,lower,upper,depth,depth,labels,defaults,edges,[1.,0.,3.],[1.,1.,1.],['POR','PERM','SW'])
        model=Toy();before=torch.get_rng_state().clone();modes=[m.training for m in model.modules()]
        probe.record(model,torch.zeros(n,1),1);assert torch.equal(before,torch.get_rng_state());assert modes==[m.training for m in model.modules()]
        probe.record(model,torch.zeros(n,1),2);report=probe.finalize();assert report['records']==2 and len(probe.raw)==int(labeled.sum())
        assert np.all(probe.history['cdf_l1_change'][1]==0)
        assert all(row['forgetting_transitions']==0 for row in report['channels'])
    return {'passed':True,'mixture_nll_independent':True,'entropy_independent_numpy':True,'default_density_gradient_zero':True,'gate_classification_mask':True,'conditional_mean_utility_identity':True,'forgetting_transition_count':True,'fixed_rows_persistence_rng_and_modes':True,'target_cdf_independent_scipy_and_endpoints':True}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))

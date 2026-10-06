"""All valid training rows, fixed inference cores, bounded CDF working memory."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from multiscale_sequence import window_indices
from por_t3_patch_loss import point_statistics

class TrainingDynamicsProbe:
    def __init__(self,out_dir,train_wells,starts,labeled,lower,upper,depth,absolute,labels,defaults,density_edges,regression_median,regression_scale,target_names):
        self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=True)
        self.device=depth.device;self.lower=lower;self.upper=upper;self.depth=depth;self.absolute=absolute;self.labels=labels;self.defaults=defaults;self.labeled=labeled
        self.edges=density_edges;self.median=np.asarray(regression_median);self.scale=np.asarray(regression_scale);self.names=list(target_names)
        self.starts=np.asarray(starts);self.train_wells=set(map(int,train_wells));assert len(self.train_wells)==64
        centers=[];anchors=[];self.window_wells=[]
        for wi in sorted(self.train_wells):
            start,end=int(starts[wi]),int(starts[wi+1]);grid=np.arange(start,end,128)+64
            chosen=np.arange(len(grid))
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
        modes=[(m,m.training) for m in model.modules()];chunks=[];seen=[];offset=0
        first=not self.epochs
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
                    current=cdf.cpu().numpy();n=len(ids)
                    if self.previous_cdf is None:
                        self.previous_cdf=np.memmap(self.out/'previous_cdf.tmp',mode='w+',dtype=np.float32,shape=(len(self.raw),)+current.shape[1:])
                    change=np.full((n,3),np.nan,np.float32) if first else np.abs(current-self.previous_cdf[offset:offset+n]).mean(-1)
                    self.previous_cdf[offset:offset+n]=current;offset+=n
                    kept={k:stats[k].cpu().numpy() for k in ['nll','conditional_entropy','target_tail','gate_probability']}
                    kept['cdf_l1_change']=change;chunks.append(kept);seen.extend(ids.cpu().tolist())
        finally:
            for m,training in modes:m.training=training
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        assert np.array_equal(self.raw,np.asarray(seen)) and offset==len(self.raw)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        for key in chunks[0]:self.history.setdefault(key,[]).append(np.concatenate([c[key] for c in chunks]).astype(np.float32))
        self.epochs.append(epoch)
        seconds=time.perf_counter()-started;self.probe_seconds+=seconds
        if self.device.type=='cuda':self.probe_peak_gpu_mib=max(self.probe_peak_gpu_mib,torch.cuda.max_memory_allocated(self.device)/1024**2)
        result={'epoch':epoch,'points':len(self.raw),'training_wells_only':True,'seconds':seconds,'total_probe_seconds':self.probe_seconds,'cumulative_gpu_peak_mib_no_reset':self.probe_peak_gpu_mib}
        print('PROBE_DONE',json.dumps(result),flush=True)
        (self.out/'training_dynamics_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

    def finalize(self):
        if not self.epochs:return {'records':0}
        if self.previous_cdf is not None:
            self.previous_cdf.flush();self.previous_cdf._mmap.close();self.previous_cdf=None
            (self.out/'previous_cdf.tmp').unlink()
        np.savez_compressed(self.out/'training_dynamics_history.npz',epochs=np.asarray(self.epochs),raw_ids=self.raw,well_ids=self.well,block_ids=self.block,default=self.default,**{k:np.stack(v) for k,v in self.history.items()})
        result={'records':len(self.epochs),'points':len(self.raw),'total_probe_seconds':self.probe_seconds,'scope':'All labeled training points, each once per observation; defaults excluded only from continuous conflict evidence. CDF temporary memory map removed at finalization.'}
        (self.out/'training_dynamics_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

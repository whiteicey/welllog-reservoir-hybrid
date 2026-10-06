"""Single-backbone-pass full-row diagnostics for member0/member1/merged Q."""
import json,time
import numpy as np
import torch
from training_por_t3_full import TrainingDynamicsProbe as BaselineProbe
from multiscale_sequence import window_indices
from por_t3_patch_loss import point_statistics

ROLES=('member0','member1','merged')
KEYS=('nll','conditional_entropy','target_tail','gate_probability','cdf_l1_change')


class TrainingDynamicsProbe(BaselineProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous={role:None for role in ROLES}
        self.latest={}
        for role in ROLES:
            folder=self.out/role;folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'population.npz',raw_ids=self.raw,well_ids=self.well,
                                block_ids=self.block,default=self.default)

    def record(self,model,x,epoch):
        started=time.perf_counter()
        assert not self.epochs or epoch>self.epochs[-1]
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None
        modes=[(module,module.training) for module in model.modules()]
        self.latest={role:{key:np.empty((len(self.raw),3),np.float32) for key in KEYS} for role in ROLES}
        offset=0;first=not self.epochs
        try:
            model.eval()
            with torch.no_grad():
                for begin in range(0,len(self.centers),16):
                    anchor=self.anchors[begin:begin+16]
                    ix,exists=window_indices(self.centers[begin:begin+16],self.lower,self.upper,anchors=anchor)
                    relative=torch.tanh((self.depth[ix]-self.depth[anchor,None])/10)
                    with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                        members,_,merged=model.prediction_parts(x[ix],relative,exists,absolute_depth=self.absolute[ix])
                    core=ix[:,64:192];use=exists[:,64:192]&self.labeled[core];ids=core[use]
                    n=len(ids)
                    assert np.array_equal(ids.cpu().numpy(),self.raw[offset:offset+n])
                    for role,(gate,raw) in zip(ROLES,[members[0],members[1],merged]):
                        stats,cdf=point_statistics(gate[:,64:192][use],raw[:,64:192][use],self.labels[ids],self.defaults[ids],self.edges)
                        current=cdf.cpu().numpy()
                        if self.previous[role] is None:
                            self.previous[role]=np.memmap(self.out/role/'previous_cdf.tmp',mode='w+',dtype=np.float32,shape=(len(self.raw),)+current.shape[1:])
                        previous=self.previous[role]
                        change=np.full((n,3),np.nan,np.float32) if first else np.abs(current-previous[offset:offset+n]).mean(-1)
                        previous[offset:offset+n]=current
                        for key in KEYS[:-1]:self.latest[role][key][offset:offset+n]=stats[key].cpu().numpy()
                        self.latest[role]['cdf_l1_change'][offset:offset+n]=change
                        del stats,cdf,current,change
                    offset+=n
                    del members,merged
        finally:
            for module,training in modes:module.training=training
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        assert offset==len(self.raw) and torch.equal(cpu_rng,torch.get_rng_state())
        if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        self.epochs.append(epoch)
        for role in ROLES:
            np.savez_compressed(self.out/role/f'record_{epoch:03d}.npz',epoch=epoch,**self.latest[role])
            self.previous[role].flush()
        seconds=time.perf_counter()-started;self.probe_seconds+=seconds
        if self.device.type=='cuda':self.probe_peak_gpu_mib=max(self.probe_peak_gpu_mib,torch.cuda.max_memory_allocated(self.device)/1024**2)
        result={'epoch':epoch,'points_per_role':len(self.raw),'roles':list(ROLES),'training_wells_only':True,
                'seconds':seconds,'total_probe_seconds':self.probe_seconds,'single_member_pair_forward_per_batch':True,
                'cdf_scratch_bytes':len(self.raw)*3*256*4*3,'history_storage':'per-role per-epoch compressed shards; latest only in RAM',
                'cumulative_gpu_peak_mib_no_reset':self.probe_peak_gpu_mib}
        print('PROBE_DONE',json.dumps(result),flush=True)
        (self.out/'training_dynamics_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

    def finalize(self):
        for role,previous in self.previous.items():
            if previous is not None:
                previous.flush();previous._mmap.close();self.previous[role]=None
                (self.out/role/'previous_cdf.tmp').unlink()
        result={'records':len(self.epochs),'epochs':self.epochs,'points_per_role':len(self.raw),
                'roles':list(ROLES),'total_probe_seconds':self.probe_seconds,
                'scope':'Full train-only population shared by three separately indexed evidence records; no validation labels, no repeated expert backbone forward.'}
        (self.out/'training_dynamics_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result

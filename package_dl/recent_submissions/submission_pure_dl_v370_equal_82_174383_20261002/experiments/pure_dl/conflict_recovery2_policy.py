"""Training-probe suspected conditional conflicts, NOT confirmed label errors.
Policy emits channel evidence weights; current global-point trainer applies minimum to complete point loss.
No automatic whole-well/block penalty; aggregation belongs to reporting only.
All arrays are [n_points,3]; missing/default channels are excluded by valid.
Thresholds are computed on currently observed valid training probes only.
"""
import numpy as np
class PersistentConflictPolicy:
    def __init__(self,n_points):
        self.n=int(n_points);self.q=np.ones((self.n,3));self.bad=np.zeros((self.n,3),bool);self.streak=np.zeros((self.n,3),np.int16);self.recovery=np.zeros((self.n,3),np.int16);self.epoch=None;self.thresholds={};self.observed=np.zeros((self.n,3),bool)
    def update(self,epoch,nll,entropy,tail,gate_probability,cdf_change,valid):
        if self.epoch is not None and epoch<=self.epoch:raise ValueError('Epoch must increase')
        arrays=[np.asarray(a,float) for a in (nll,entropy,tail,gate_probability,cdf_change)];v=np.asarray(valid,bool)
        assert v.shape==(self.n,3) and all(a.shape==v.shape for a in arrays)
        loss,ent,t,pg,change=arrays;observed=v&np.logical_and.reduce([np.isfinite(a) for a in arrays]);self.observed=observed;self.epoch=int(epoch)
        self.streak[~observed]=0;self.recovery[~observed]=0
        self.thresholds={'nll_p90':[],'nll_p75':[],'entropy_p40':[],'cdf_change_median':[]}
        triggered=np.zeros_like(v);recoverable=np.zeros_like(v)
        for c in range(3):
            use=observed[:,c]
            if not use.any():
                for values in self.thresholds.values():values.append(None)
                continue
            p90,p75=np.quantile(loss[use,c],[.9,.75]);p25=np.quantile(ent[use,c],.40);median=np.median(change[use,c])
            for key,value in [('nll_p90',p90),('nll_p75',p75),('entropy_p40',p25),('cdf_change_median',median)]:self.thresholds[key].append(float(value))
            if epoch<10:continue
            triggered[:,c]=use&(loss[:,c]>=p90)&(ent[:,c]<=p25)&(t[:,c]<.02)&(change[:,c]<=median)&(pg[:,c]<.5)
            recoverable[:,c]=use&(t[:,c]>=.1)&(loss[:,c]<=p75)
        self.streak=np.where(triggered,self.streak+1,0).astype(np.int16)
        newly_bad=(self.streak==3)
        self.bad|=newly_bad;self.q[newly_bad]=.2
        # Two currently bad channels share a stronger initial contribution reduction.
        # Apply on a newly completed three-record conflict event only, so recovery is not undone every record.
        multi=(self.bad.sum(1)>=2)&newly_bad.any(1)
        self.q[multi[:,None]&self.bad&observed]=.1
        self.recovery=np.where(recoverable&self.bad,self.recovery+1,0).astype(np.int16)
        restore=(self.recovery>=2)&self.bad
        self.q[restore]=np.minimum(1.,self.q[restore]*2);self.recovery[restore]=0
        cleared=self.bad&(self.q>=1);self.bad[cleared]=False;self.streak[cleared]=0
        # Missing observations never create new penalties or recover existing ones.
        assert ((self.q>=.1)&(self.q<=1)).all()
        return self.q.copy()
    def snapshot(self):
        return {'epoch':self.epoch,'thresholds':self.thresholds,'downweighted_by_channel':(self.q<1).sum(0).tolist(),'suspected_bad_by_channel':self.bad.sum(0).tolist(),'observed_by_channel':self.observed.sum(0).tolist(),'max_conflict_streak_by_channel':self.streak.max(0).tolist() if self.n else [0]*3,'max_recovery_streak_by_channel':self.recovery.max(0).tolist() if self.n else [0]*3,'weight_range':[float(self.q.min()),float(self.q.max())] if self.n else [1.,1.],'interpretation':'Suspected persistent conditional conflict, not confirmed target error. Channel evidence weights are reduced to a point weight by the global-point trainer. No whole-well/block punishment. Multi-channel reduction applied on newly completed three-record conflict events, only observed channels; later sustained recovery can rise above0.1.'}

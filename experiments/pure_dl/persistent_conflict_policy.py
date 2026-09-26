"""Training-probe suspected conditional conflicts, NOT confirmed label errors.
Only continuous NLL is eligible for Q; gate BCE must remain unweighted.
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
        self.thresholds={'nll_p90':[],'nll_p75':[],'entropy_p25':[],'cdf_change_median':[]}
        triggered=np.zeros_like(v);recoverable=np.zeros_like(v)
        for c in range(3):
            use=observed[:,c]
            if not use.any():
                for values in self.thresholds.values():values.append(None)
                continue
            p90,p75=np.quantile(loss[use,c],[.9,.75]);p25=np.quantile(ent[use,c],.25);median=np.median(change[use,c])
            for key,value in [('nll_p90',p90),('nll_p75',p75),('entropy_p25',p25),('cdf_change_median',median)]:self.thresholds[key].append(float(value))
            if epoch<10:continue
            triggered[:,c]=use&(loss[:,c]>=p90)&(ent[:,c]<=p25)&(t[:,c]<.01)&(change[:,c]<=median)&(pg[:,c]<.5)
            recoverable[:,c]=use&(t[:,c]>=.1)&(loss[:,c]<=p75)
        self.streak=np.where(triggered,self.streak+1,0).astype(np.int16)
        newly_bad=(self.streak==3)
        self.bad|=newly_bad;self.q[newly_bad]=.2
        # Two currently bad channels share a stronger initial contribution reduction.
        # Apply on a newly completed three-record conflict event only, so recovery is not undone every record.
        multi=(self.bad.sum(1)>=2)&newly_bad.any(1)
        self.q[multi[:,None]&self.bad&observed]=.1
        self.recovery=np.where(recoverable&self.bad,self.recovery+1,0).astype(np.int16)
        restore=(self.recovery>=3)&self.bad
        self.q[restore]=np.minimum(1.,self.q[restore]*2);self.recovery[restore]=0
        cleared=self.bad&(self.q>=1);self.bad[cleared]=False;self.streak[cleared]=0
        # Missing observations never create new penalties or recover existing ones.
        assert ((self.q>=.1)&(self.q<=1)).all()
        return self.q.copy()
    def snapshot(self):
        return {'epoch':self.epoch,'thresholds':self.thresholds,'downweighted_by_channel':(self.q<1).sum(0).tolist(),'suspected_bad_by_channel':self.bad.sum(0).tolist(),'observed_by_channel':self.observed.sum(0).tolist(),'max_conflict_streak_by_channel':self.streak.max(0).tolist() if self.n else [0]*3,'max_recovery_streak_by_channel':self.recovery.max(0).tolist() if self.n else [0]*3,'weight_range':[float(self.q.min()),float(self.q.max())] if self.n else [1.,1.],'interpretation':'Suspected persistent conditional conflict, not confirmed target error. Q applies only to continuous NLL; gate BCE unchanged. No whole-well/block punishment. Multi-channel reduction applied on newly completed three-record conflict events, only observed channels; later sustained recovery can rise above0.1.'}

def self_test():
    n=100;loss=np.ones((n,3));ent=np.full((n,3),.5);tail=np.full((n,3),.4);pg=np.full((n,3),.1);change=np.full((n,3),.1);valid=np.ones((n,3),bool)
    loss[0,:2]=10;ent[0,:2]=.01;tail[0,:2]=.001;change[0,:2]=.01
    # Large-loss but high entropy point is hard/uncertain and must remain neutral.
    loss[1]=20;ent[1]=.9;tail[1]=.001
    policy=PersistentConflictPolicy(n)
    for e in [2,4,6,8]:assert np.all(policy.update(e,loss,ent,tail,pg,change,valid)==1)
    policy.update(10,loss,ent,tail,pg,change,valid);policy.update(12,loss,ent,tail,pg,change,valid)
    q=policy.update(14,loss,ent,tail,pg,change,valid);assert np.allclose(q[0],[.1,.1,1]) and np.all(q[1:]==1)
    loss[0,:2]=.5;tail[0,:2]=.5
    for e in [16,18,20]:q=policy.update(e,loss,ent,tail,pg,change,valid)
    assert np.allclose(q[0],[.2,.2,1])
    for e in range(22,40,2):q=policy.update(e,loss,ent,tail,pg,change,valid)
    assert np.all(q==1) and not policy.bad.any()
    # Missing/default evidence breaks streak, does not flag other channels.
    single=PersistentConflictPolicy(n);loss[0,0]=10;tail[0,0]=.001
    single.update(10,loss,ent,tail,pg,change,valid);missing=valid.copy();missing[0,0]=False
    assert np.all(single.update(12,loss,ent,tail,pg,change,missing)==1)
    single.update(14,loss,ent,tail,pg,change,valid);single.update(16,loss,ent,tail,pg,change,valid)
    q=single.update(18,loss,ent,tail,pg,change,valid);assert np.allclose(q[0],[.2,1,1])
    before=q.copy();single.update(20,loss,ent,tail,pg,change,np.zeros_like(valid));assert np.array_equal(before,single.q)
    return {'passed':True,'hard_uncertain_neutral':True,'persistent_conflict_only':True,'two_channel_point_01':True,'progressive_recovery':True,'default_missing_break_streak':True,'channel_isolation':True,'not_anomaly_proof':True}
if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))

"""EXPERIMENTAL_NOT_TRAINED: observation-only hierarchical priority A, not trust Q.

API::
    tracker = HierarchicalPriority()
    weights = tracker.update(epoch, point_ids, well_ids, block_ids, losses)
    weights = tracker.weights(point_ids, well_ids, block_ids)
    summary = tracker.snapshot()

Inputs are aligned one-dimensional integer IDs and [N,3] nonnegative difficulty;
NaN means unobserved. Duplicate point IDs per update and changed point membership
are rejected. block identity is (well_id, block_id), not block_id alone. Call once
per distinct, increasing epoch with fixed-probe observations; no training occurs.

Per channel, clipping tau is the first available p95 at epoch>=10, frozen forever.
Clipping affects diagnostic statistics ONLY. Prior to freezing there is no clip.
At per-channel tau freeze, that channel EMA/history resets so clipping itself
cannot be mistaken for learning progress; five new record intervals are needed.
Current point difficulties are equally averaged within each observed block, then
observed block means equally averaged within each well (wells are NOT row-weighted).
Each level independently applies EMA new=.8*old+.2*current, first value=current.
At a level/channel, h=clip((EMA-p50)/(p90-p50+eps),0,1), among currently observed
units. g=clip(((EMA_5_records_ago-EMA)/(abs(EMA_5_records_ago)+eps))/.05,0,1).
The lag is five UPDATE RECORDS, not five epochs, and requires that exact record
and the current channel be observed. Missing/warmup/insufficient-history units
receive 1. A=1+.5*h*g. Returned priority is the geometric mean of point/block/well
A. It is in [1,1.5], never downweights, and is never renormalized.

No validation, targets, anomaly flags, sample trust Q or feature filtering API.
A caller may transform negative NLL to nonnegative softplus(NLL) for this probe;
this module neither computes nor changes the actual training loss. Nine-curve
quality assessment Q remains independent and is intentionally absent here.
"""
import numpy as np

STATUS='EXPERIMENTAL_NOT_TRAINED'

class HierarchicalPriority:
    def __init__(self):
        self.tau=np.full(3,np.nan);self.tau_epoch=[None]*3;self.epoch=None;self.record=-1
        self.states={level:{} for level in ('point','block','well')};self.membership={};self.last_keys={level:set() for level in self.states}
    @staticmethod
    def _ids(values,n=None):
        a=np.asarray(values)
        if a.ndim!=1 or a.dtype.kind not in 'iu':raise ValueError('IDs must be one-dimensional integers')
        if n is not None and len(a)!=n:raise ValueError('ID lengths differ')
        return [int(x) for x in a]
    @staticmethod
    def _mean(rows):
        rows=np.stack(rows);valid=np.isfinite(rows);count=valid.sum(0)
        return np.divide(np.where(valid,rows,0).sum(0),count,out=np.full(3,np.nan),where=count>0)
    def update(self,epoch,point_ids,well_ids,block_ids,losses):
        points=self._ids(point_ids);wells=self._ids(well_ids,len(points));blocks=self._ids(block_ids,len(points))
        if len(set(points))!=len(points):raise ValueError('Duplicate point IDs in one record')
        if self.epoch is not None and epoch<=self.epoch:raise ValueError('Epochs must increase')
        loss=np.asarray(losses,dtype=np.float64)
        if loss.shape!=(len(points),3):raise ValueError('losses must have shape [N,3]')
        if np.isinf(loss).any() or (loss[np.isfinite(loss)]<0).any():raise ValueError('Difficulty must be finite nonnegative or NaN')
        for p,w,b in zip(points,wells,blocks):
            if p in self.membership and self.membership[p]!=(w,b):raise ValueError('Point membership changed')
        self.epoch=int(epoch);self.record+=1
        for p,w,b in zip(points,wells,blocks):self.membership[p]=(w,b)
        for c in range(3):
            v=loss[:,c];v=v[np.isfinite(v)]
            if epoch>=10 and np.isnan(self.tau[c]) and len(v):
                self.tau[c]=np.quantile(v,.95);self.tau_epoch[c]=int(epoch)
                # A new clipping regime must not masquerade as model improvement.
                for states in self.states.values():
                    for state in states.values():
                        state['ema'][c]=np.nan
                        for previous in state['history'].values():previous[c]=np.nan
        clipped=np.minimum(loss,np.where(np.isfinite(self.tau),self.tau,np.inf))
        point_current={p:clipped[i].copy() for i,p in enumerate(points)}
        children={}
        # Sort IDs so aggregation is invariant to input row order, including rounding.
        for p in sorted(points):children.setdefault(self.membership[p],[]).append(point_current[p])
        block_current={key:self._mean(rows) for key,rows in children.items()}
        children={}
        for (w,b),value in sorted(block_current.items()):children.setdefault(w,[]).append(value)
        well_current={w:self._mean(rows) for w,rows in children.items()}
        for level,current in [('point',point_current),('block',block_current),('well',well_current)]:
            states=self.states[level];self.last_keys[level]=set(current)
            for key,value in current.items():
                state=states.setdefault(key,{'ema':np.full(3,np.nan),'history':{},'priority':np.ones(3),'observed':np.zeros(3,bool)})
                observed=np.isfinite(value);old=state['ema'];new=np.where(np.isfinite(old),.8*old+.2*value,value)
                state['ema']=np.where(observed,new,old);state['observed']=observed;state['priority']=np.ones(3)
                state['history'][self.record]=np.where(observed,state['ema'],np.nan)
                state['history']={i:v for i,v in state['history'].items() if i>=self.record-5}
            for c in range(3):
                keys=sorted(k for k in current if states[k]['observed'][c])
                if not keys or epoch<10:continue
                q50,q90=np.quantile([states[k]['ema'][c] for k in keys],[.5,.9])
                for key in keys:
                    state=states[key];now=state['ema'][c];past=state['history'].get(self.record-5,np.full(3,np.nan))[c]
                    if not np.isfinite(past):continue
                    h=np.clip((now-q50)/(q90-q50+1e-12),0,1)
                    drop=past-now
                    g=np.clip(drop/(abs(past)+1e-12)/.05,0,1) if drop>1e-12*(1+abs(past)) else 0.
                    state['priority'][c]=1+.5*h*g
        return self.weights(points,wells,blocks)
    def weights(self,point_ids,well_ids,block_ids):
        points=self._ids(np.asarray(point_ids,dtype=np.int64));wells=self._ids(np.asarray(well_ids,dtype=np.int64),len(points));blocks=self._ids(np.asarray(block_ids,dtype=np.int64),len(points))
        answer=np.ones((len(points),3))
        for i,(p,w,b) in enumerate(zip(points,wells,blocks)):
            if self.membership.get(p)!=(w,b) or p not in self.last_keys['point']:continue
            known=self.states['point'][p]['observed'];product=np.ones(3)
            for level,key in [('point',p),('block',(w,b)),('well',w)]:
                if key not in self.last_keys[level]:continue
                product*=self.states[level][key]['priority']
            answer[i]=np.where(known,np.clip(np.cbrt(product),1,1.5),1.)
        return answer
    def snapshot(self):
        layers={}
        for level,states in self.states.items():
            keys=sorted(self.last_keys[level]);a=np.stack([states[k]['priority'] for k in keys]) if keys else np.ones((0,3))
            layers[level]={'currently_observed_units':len(keys),'increased_by_channel':(a>1).sum(0).tolist(),'range':[float(a.min()),float(a.max())] if a.size else [1.,1.]}
        return {'status':STATUS,'epoch':self.epoch,'record_index':self.record,'warmup_until_epoch':10,'tau':[float(t) if np.isfinite(t) else None for t in self.tau],'tau_frozen_epoch':self.tau_epoch,'layers':layers,'priority_range':[1.,1.5],'trust_Q_implemented':False,'statistics_only_clipping':True,'progress_lag':'five recorded updates','aggregation':'point losses equally within blocks; block means equally within wells; independent EMA at each level'}

def self_test():
    p=np.arange(20);w=np.repeat(np.arange(4),5);b=np.tile([0,0,0,1,1],4)
    fixed=np.tile(np.linspace(1,20,20)[:,None],(1,3));tracker=HierarchicalPriority()
    for epoch in range(1,18):a=tracker.update(epoch,p,w,b,fixed)
    assert np.all(a==1),'Unimproved hard points must stay neutral'
    assert np.all(tracker.weights([999],[0],[0])==1)
    assert len(tracker.states['block'])==8,'Same block IDs from different wells must remain separate'
    improving=HierarchicalPriority();reordered=HierarchicalPriority()
    for epoch in range(1,19):
        loss=fixed.copy();loss[10:]*=(1-.018*epoch)
        a=improving.update(epoch,p,w,b,loss);order=np.arange(19,-1,-1);z=reordered.update(epoch,p[order],w[order],b[order],loss[order])
        assert np.array_equal(a,z[order]);assert np.all((a>=1)&(a<=1.5))
        if epoch<10:assert np.all(a==1)
    assert np.any(a>1),'Improving hard samples should increase'
    tau=improving.tau.copy();improving.update(19,p,w,b,fixed*100);assert np.array_equal(tau,improving.tau)
    missing=HierarchicalPriority();full=HierarchicalPriority()
    for epoch in range(1,18):
        loss=fixed*(1-.015*epoch);a=full.update(epoch,p,w,b,loss);loss[:,1]=np.nan;z=missing.update(epoch,p,w,b,loss)
        assert np.array_equal(a[:,[0,2]],z[:,[0,2]]) and np.all(z[:,1]==1)
    try:tracker.update(18,[1,1],[0,0],[0,0],np.ones((2,3)))
    except ValueError:pass
    else:raise AssertionError('Duplicate points accepted')
    try:tracker.update(18,[1],[0],[0],np.array([[-1.,0,0]]))
    except ValueError:pass
    else:raise AssertionError('Negative difficulty accepted')
    # Equal block weighting: block sizes three and two must not change the well mean.
    agg=HierarchicalPriority();v=np.array([[2.,2,2],[2,2,2],[2,2,2],[10,10,10],[10,10,10]])
    agg.update(1,np.arange(5),np.zeros(5,int),np.array([0,0,0,1,1]),v)
    assert np.all(agg.states['well'][0]['ema']==6)
    return {'passed':True,'status':STATUS,'bounded_priority':True,'unknown_neutral':True,'no_progress_neutral':True,'improving_hard_increased':True,'input_order_invariance':True,'cross_well_block_isolation':True,'duplicates_and_negative_rejected':True,'channel_missingness_isolated':True,'equal_child_aggregation':True,'tau_frozen':True}
if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))

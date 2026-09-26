"""EXPERIMENTAL_NOT_TRAINED. Mild difficulty downweighting with separate recoverable input-quality evidence.
A_layer=1+0.5*h*g-0.2*h*(1-g), range[0.8,1.5]; no progress is not an anomaly verdict.
Point/block/well A combine geometrically, unknown or missing observations are neutral.
Quality base uses strongest input flag evidence, not repeated multiplication; R below is an engineering contribution weight, not target correctness probability.
"""
import numpy as np

STATUS='EXPERIMENTAL_NOT_TRAINED_RECOVERABLE'

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
                    state['priority'][c]=1+.5*h*g-.2*h*(1-g)
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
            answer[i]=np.where(known,np.clip(np.cbrt(product),.8,1.5),1.)
        return answer
    def snapshot(self):
        layers={}
        for level,states in self.states.items():
            keys=sorted(self.last_keys[level]);a=np.stack([states[k]['priority'] for k in keys]) if keys else np.ones((0,3))
            layers[level]={'currently_observed_units':len(keys),'increased_by_channel':(a>1).sum(0).tolist(),'range':[float(a.min()),float(a.max())] if a.size else [1.,1.]}
        return {'status':STATUS,'epoch':self.epoch,'record_index':self.record,'warmup_until_epoch':10,'tau':[float(t) if np.isfinite(t) else None for t in self.tau],'tau_frozen_epoch':self.tau_epoch,'layers':layers,'priority_range':[.8,1.5],'trust_Q_implemented':False,'statistics_only_clipping':True,'progress_lag':'five recorded updates','aggregation':'point losses equally within blocks; block means equally within wells; independent EMA at each level'}


class RecoverableInputQuality:
    """Fixed training probe IDs only. Pass channel-valid mask so missing supervision never updates quality.
    Loss is nonnegative unweighted diagnostic; changes is adjacent-probe CDF difference.
    Three consecutive >=5% loss improvements and <=training-channel median CDF change
    recover by factor1.5 toward1. A >=5% loss increase with above-median change relapses
    by factor0.5, never below the fixed input-quality base. It does not repair input data.
    """
    def __init__(self, flagged_count, block_flag_fraction, well_flag_fraction):
        k=np.asarray(flagged_count);b=np.asarray(block_flag_fraction,float);w=np.asarray(well_flag_fraction,float)
        assert k.ndim==1 and b.shape==w.shape==k.shape and (k>=0).all()
        assert ((b>=0)&(b<=1)&(w>=0)&(w<=1)).all()
        point=np.where(k>=2,.1,np.where(k==1,.2,1.))
        quality=np.minimum(point,np.minimum(np.where(b>=.25,.5,1.),np.where(w>=.10,.8,1.)))
        self.base=np.repeat(quality[:,None],3,1);self.value=self.base.copy();self.streak=np.zeros_like(self.value,dtype=np.int16);self.previous=None
    def update(self, losses, cdf_changes, valid):
        loss=np.asarray(losses,float);change=np.asarray(cdf_changes,float);valid=np.asarray(valid,bool)
        assert loss.shape==change.shape==valid.shape==self.base.shape
        observed=valid&np.isfinite(loss)&(loss>=0)&np.isfinite(change)
        if self.previous is not None:
            for c in range(3):
                obs=observed[:,c]&np.isfinite(self.previous[:,c])
                if not obs.any():self.streak[:,c]=0;continue
                cutoff=np.median(change[obs,c])
                improved=obs&(loss[:,c]<=.95*self.previous[:,c])&(change[:,c]<=cutoff)
                worse=obs&(loss[:,c]>=1.05*self.previous[:,c])&(change[:,c]>cutoff)
                self.streak[:,c]=np.where(improved,self.streak[:,c]+1,0)
                restore=self.streak[:,c]>=3
                self.value[restore,c]=np.minimum(1,self.value[restore,c]*1.5);self.streak[restore,c]=0
                self.value[worse,c]=np.maximum(self.base[worse,c],self.value[worse,c]*.5)
        self.previous=np.where(valid&np.isfinite(loss),loss,np.nan)
        assert (self.value>=self.base).all() and (self.value<=1).all()
        return self.value.copy()

def self_test():
    p=np.arange(20);w=np.repeat(np.arange(4),5);b=np.tile([0,0,0,1,1],4)
    loss=np.tile(np.linspace(1,20,20)[:,None],(1,3));t=HierarchicalPriority()
    for e in range(1,18):a=t.update(e,p,w,b,loss)
    assert a.min()>=.8 and a.min()<1 and a.max()<=1.5
    assert np.array_equal(t.weights([999],[9],[9]),np.ones((1,3)))
    q=RecoverableInputQuality(np.array([1,2,0,0]),np.array([0.,0.,.3,0.]),np.array([0.,0.,0.,.2]))
    assert np.allclose(q.base[:,0],[.2,.1,.5,.8])
    value=np.full((4,3),10.);cdf=np.tile(np.array([.01,.02,.03,.04])[:,None],(1,3));valid=np.ones((4,3),bool)
    q.update(value,cdf,valid)
    for _ in range(3):value*=.9;q.update(value,cdf,valid)
    assert np.allclose(q.value[:2,0],[.3,.15]) and np.allclose(q.value[2:,0],[.5,.8])
    before=q.value.copy();q.update(value*1.2,cdf[::-1],valid)
    assert np.allclose(q.value[:2,0],[.2,.1]) and (q.value>=q.base).all()
    old=q.value.copy();q.update(value*.5,cdf,np.zeros_like(valid));assert np.array_equal(q.value,old)
    assert np.allclose(RecoverableInputQuality([2],[1.],[1.]).base,.1), 'No repeated evidence penalty'
    assert np.allclose(np.clip(np.array([.8,1.5])*np.array([.1,1]),.1,1.5),[.1,1.5])
    return {'passed':True,'priority_range':[.8,1.5],'quality_base_min':.1,'recovery_and_relapse_checked':True,'no_training':True}
if __name__=='__main__':
    import json
    print(json.dumps(self_test()))

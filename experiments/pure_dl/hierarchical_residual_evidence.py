"""Train-only relative residual evidence; protects coherent runs, not proof of errors."""
import numpy as np


class HierarchicalResidualEvidence:
    def __init__(self, raw, well, block):
        self.raw=np.asarray(raw);self.well=np.asarray(well);self.block=np.asarray(block)
        assert self.raw.ndim==1 and self.raw.shape==self.well.shape==self.block.shape
        assert np.all(np.diff(self.raw)>0)
        self.well_groups=[np.flatnonzero(self.well==w) for w in np.unique(self.well)]
        keys=np.stack([self.well,self.block],axis=1)
        split=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1,len(keys)]
        self.block_groups=[slice(int(a),int(b)) for a,b in zip(split[:-1],split[1:])]

    @staticmethod
    def location_scale(x):
        center=np.median(x)
        return center,max(float(1.4826*np.median(np.abs(x-center))),.1)

    def compute(self, residual, valid):
        residual=np.asarray(residual,float);valid=np.asarray(valid,bool)&np.isfinite(residual)
        assert residual.shape==valid.shape==(len(self.raw),3)
        z=np.zeros_like(residual);outlier=np.zeros_like(valid)
        for c in range(3):
            center=np.zeros(len(self.raw));scale=np.ones(len(self.raw))
            for ix in self.well_groups:
                values=residual[ix,c][valid[ix,c]]
                if len(values):center[ix],scale[ix]=self.location_scale(values)
            for group in self.block_groups:
                local=valid[group,c];values=residual[group,c][local]
                if len(values)>=16:
                    m,s=self.location_scale(values);a=len(values)/(len(values)+32.)
                    center[group]=a*m+(1-a)*center[group]
                    scale[group]=a*s+(1-a)*scale[group]
            z[:,c]=(residual[:,c]-center)/scale
            outlier[:,c]=valid[:,c]&(np.abs(z[:,c])>3)
        edge=(np.diff(self.raw)==1)&(self.well[1:]==self.well[:-1])
        same=edge[:,None]&outlier[1:]&outlier[:-1]&(np.sign(z[1:])==np.sign(z[:-1]))
        coherent=np.zeros_like(valid)
        if len(self.raw)>=3:
            triple=same[:-1]&same[1:]
            coherent[:-2]|=triple;coherent[1:-1]|=triple;coherent[2:]|=triple
        isolated=outlier&~coherent
        return isolated,{'relative_outlier_by_channel':outlier.sum(0).tolist(),
                         'coherent_run_protected_by_channel':coherent.sum(0).tolist(),
                         'isolated_candidate_by_channel':isolated.sum(0).tolist()}


if __name__=='__main__':
    raw=np.arange(256);well=raw//128;block=raw//64
    evidence=HierarchicalResidualEvidence(raw,well,block)
    r=np.zeros((256,3));v=np.ones_like(r,dtype=bool)
    r[30,0]=10;r[60:65,1]=10;r[128:,2]=8
    isolated,stats=evidence.compute(r,v)
    assert isolated[30,0] and isolated[:,0].sum()==1
    assert not isolated[60:65,1].any() and not isolated[:,2].any()
    # Separate wells never form a protected contiguous three-point run.
    r[:]=0;r[127:130,0]=10
    isolated,_=evidence.compute(r,v);assert isolated[127:130,0].all()
    # A missing/default label breaks geological continuity evidence.
    r[:]=0;r[40:43,0]=10;v[41,0]=False
    isolated,_=evidence.compute(r,v);assert isolated[40,0] and isolated[42,0]
    print('PASS isolated spike, coherent run across block edge, whole-well shift, well boundary and missing-label isolation')

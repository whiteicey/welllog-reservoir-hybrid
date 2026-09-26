"""Same-well 128-row input-only patch memory; never accepts target arrays."""
import numpy as np
import torch

CURVES=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']


class ExternalWellMemory:
    def __init__(self, rec, starts, x, names, depth):
        self.device=x.device;starts=np.asarray(starts);lengths=np.diff(starts)
        count=(lengths+127)//128;maximum=int(count.max());nw=len(rec)
        features=np.zeros((nw,maximum,30),np.float32)
        centers=np.zeros((nw,maximum),np.float32)
        lower=np.zeros((nw,maximum),np.int64);upper=lower.copy();valid=np.zeros_like(lower,bool)
        columns=[names.index(name) for name in CURVES]
        for w,(_,frame) in enumerate(rec):
            n=len(frame);blocks=int(count[w]);padded=blocks*128
            values=x[int(starts[w]):int(starts[w+1]),columns].float().cpu().numpy()
            observed=np.isfinite(frame[CURVES].to_numpy(float))
            dep=depth[int(starts[w]):int(starts[w+1])].cpu().numpy()
            data=np.zeros((padded,9),np.float32);mask=np.zeros((padded,9),bool)
            data[:n]=values;mask[:n]=observed;data=data.reshape(blocks,128,9);mask=mask.reshape(blocks,128,9)
            cnt=mask.sum(1);mean=np.where(mask,data,0).sum(1)/cnt.clip(1)
            var=np.where(mask,(data-mean[:,None])**2,0).sum(1)/cnt.clip(1)
            actual=np.minimum(128,n-np.arange(blocks)*128)
            features[w,:blocks,:9]=mean;features[w,:blocks,9:18]=np.sqrt(var)
            features[w,:blocks,18:27]=1-cnt/actual[:,None]
            features[w,:blocks,27]=actual/128
            for b in range(blocks):
                a=b*128;z=min(a+128,n);centers[w,b]=dep[a:z].mean()
                lower[w,b]=starts[w]+a;upper[w,b]=starts[w]+z
                features[w,b,28]=(centers[w,b]-dep[0])/max(float(dep[-1]-dep[0]),1e-6)
                features[w,b,29]=np.log1p(abs(float(dep[z-1]-dep[a]))/10)
            valid[w,:blocks]=cnt.sum(1)>0
        self.features=torch.tensor(features,device=self.device)
        self.centers=torch.tensor(centers,device=self.device)
        self.lower=torch.tensor(lower,device=self.device);self.upper=torch.tensor(upper,device=self.device)
        self.valid=torch.tensor(valid,device=self.device)
        self.row_well=torch.tensor(np.repeat(np.arange(nw),lengths),device=self.device)
        self.depth=depth
        self.metadata={'patch_rows':128,'features':31,'curves':CURVES,'patches_per_well':count.tolist(),
                       'statistics':'train-fold normalized raw input means/std, missing fraction, fullness, relative well depth, depth span, signed relative distance',
                       'targets_used':False,'overlap_rule':'exclude entire patch when any raw row overlaps local window'}

    def gather(self, centers, anchors=None, length=256):
        anchors=centers if anchors is None else anchors
        well=self.row_well[anchors]
        outside=(self.upper[well]<=centers[:,None]-length//2)|(self.lower[well]>=centers[:,None]+length//2)
        mask=self.valid[well]&outside
        delta=self.centers[well]-self.depth[anchors,None]
        distance=delta.sign()*torch.log1p(delta.abs()/10)
        tokens=torch.cat([self.features[well],distance[...,None]],-1)
        return tokens.masked_fill(~mask[...,None],0),mask

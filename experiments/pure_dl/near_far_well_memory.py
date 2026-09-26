"""Eight nearest external 128-row patches; remaining patches pooled in groups of four."""
import torch
from torch.nn import functional as F
from external_well_memory import ExternalWellMemory


class NearFarWellMemory(ExternalWellMemory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata.update(features=32,nearest_fine_patches=8,far_group_patches=4,
                             far_span='row-weighted mean of constituent spans',
                             scale_indicator='0 near128, 1 far pooled up to512')

    def gather(self, centers, anchors=None, length=256):
        anchors=centers if anchors is None else anchors
        token,valid=super().gather(centers,anchors,length)
        batch,count,_=token.shape;k=min(8,count)
        distance=token[...,-1].abs().masked_fill(~valid,float('inf'))
        ids=distance.argsort(-1)[:,:k]
        near=torch.gather(token,1,ids[...,None].expand(-1,-1,31))
        near_valid=torch.gather(valid,1,ids)
        chosen=torch.zeros_like(valid).scatter(1,ids,near_valid)
        far_valid=valid&~chosen
        padding=(-count)%4
        fine=F.pad(token[...,:30],(0,0,0,padding)).reshape(batch,-1,4,30)
        active=F.pad(far_valid,(0,padding)).reshape(batch,-1,4)
        rows=fine[...,27]*128*active
        total=rows.sum(-1);counts=(1-fine[...,18:27])*rows[...,None]
        observed=counts.sum(-2)
        mean=(fine[...,:9]*counts).sum(-2)/observed.clamp_min(1)
        second=((fine[...,9:18].square()+fine[...,:9].square())*counts).sum(-2)/observed.clamp_min(1)
        std=(second-mean.square()).clamp_min(0).sqrt()
        missing=1-observed/total[...,None].clamp_min(1)
        relative=(fine[...,28]*rows).sum(-1)/total.clamp_min(1)
        # Mean constituent depth span, not a claim that retained patches are contiguous.
        span=(fine[...,29]*rows).sum(-1)/total.clamp_min(1)
        well=self.row_well[anchors]
        depths=F.pad(self.centers[well],(0,padding)).reshape(batch,-1,4)
        center=(depths*rows).sum(-1)/total.clamp_min(1)
        delta=center-self.depth[anchors,None]
        signed_distance=delta.sign()*torch.log1p(delta.abs()/10)
        far=torch.cat([mean,std,missing,(total/512)[...,None],relative[...,None],
                       span[...,None],signed_distance[...,None],torch.ones_like(total[...,None])],-1)
        near=torch.cat([near,torch.zeros_like(near[...,:1])],-1)
        mask=torch.cat([near_valid,total>0],1)
        return torch.cat([near,far],1).masked_fill(~mask[...,None],0),mask

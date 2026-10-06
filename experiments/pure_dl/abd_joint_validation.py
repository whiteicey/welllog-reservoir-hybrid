"""Deterministic four-window validation of the jointly merged forty-component ensemble."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
import train_invariant as base
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from global_cosine_sequence import Network as SingleNetwork
from por_t3_patch_loss import probability


class JointFold5Validation:
    def __init__(self, *, x,depth,lower,upper,absolute,starts,row_ids,raw_to_valid,
                 validation,preprocessing):
        self.x,self.depth,self.lower,self.upper,self.absolute=x,depth,lower,upper,absolute
        self.starts,self.row_ids,self.raw_to_valid=starts,row_ids,raw_to_valid
        self.validation=validation
        self.local=np.full(sum(map(len,row_ids)),-1,np.int64)
        self.local[validation]=np.arange(len(validation))
        self.wells=[i for i,rows in enumerate(row_ids) if self.local[rows[0]]>=0]
        assert len(self.wells)==16 and len(validation)==144665
        self.actions=[];self.edges=[];self.utilities=[];self.source_checks=[]
        self.core_weights=torch.sin(torch.pi*(torch.arange(128,device='cuda')+.5)/128).square()
        for j,name in enumerate(base.TS):
            ct=np.asarray(preprocessing['centers'][j])
            action=np.linspace(ct[0],ct[-1],800) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
            fine=np.linspace(ct[0],ct[-1],3197) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,3197)
            self.actions.append(np.unique(np.r_[base.DEFAULTS[name],np.clip(action,*base.BOUNDS[name]),np.clip(fine,*base.BOUNDS[name])]))
            support=np.linspace(base.TRANSFORM[j](ct[0]),base.TRANSFORM[j](ct[-1]),1024)
            edge=(np.r_[-np.inf,(support[:-1]+support[1:])/2,np.inf]-preprocessing['regression_median'][j])/preprocessing['regression_scale'][j]
            self.edges.append(torch.tensor(edge,device='cuda',dtype=torch.float32))
            utility=base.kernel_matrix(self.actions[j],np.r_[base.DEFAULTS[name],base.INVERSE[j](support)],name).T
            self.utilities.append(torch.tensor(utility,device='cuda',dtype=torch.float32))
    @torch.no_grad()
    def well_joint(self,model,wi):
        model.eval();rows=self.row_ids[wi];offset=int(rows[0])
        result=torch.zeros(len(rows),3,1025,device='cuda')
        total=torch.zeros(len(rows),device='cuda');covered=np.zeros(len(rows),np.int8)
        cc_all=np.arange(self.starts[wi]-32,self.starts[wi+1]+64,32)
        for start in range(0,len(cc_all),16):
            cc=torch.as_tensor(cc_all[start:start+16],device='cuda')
            anchors=cc.clamp(int(self.starts[wi]),int(self.starts[wi+1]-1))
            ix,exists=window_indices(cc,self.lower,self.upper,anchors=anchors)
            relative=torch.tanh((self.depth[ix]-self.depth[anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                go,raw=model(self.x[ix],relative,exists,absolute_depth=self.absolute[ix])
            ids=self.raw_to_valid[ix[:,64:192].cpu().numpy()]
            use=exists[:,64:192].cpu().numpy() & (ids>=0)
            loc=ids[use]-offset
            assert ((loc>=0)&(loc<len(rows))).all()
            np.add.at(covered,loc,1)
            select=torch.as_tensor(use,device='cuda');loc=torch.as_tensor(loc,device='cuda')
            weight=self.core_weights[None,:].expand(len(cc),-1)[select]
            total.index_add_(0,loc,weight)
            gate=go[:,64:192][select].float().sigmoid();r=raw[:,64:192][select].float()
            for j in range(3):
                cp=probability(j,r[:,j,0].softmax(-1),r[:,j,1],F.softplus(r[:,j,2])+.025,self.edges[j])
                joint=torch.cat([gate[:,j:j+1],(1-gate[:,j:j+1])*cp],-1)
                result[:,j].index_add_(0,loc,joint*weight[:,None])
        assert (covered==4).all() and torch.allclose(total,torch.full_like(total,2.),atol=1e-5)
        return result/total[:,None,None]

    @torch.no_grad()
    def evaluate(self,model):
        previous=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            pred=np.empty((len(self.validation),3),np.float64)
            for wi in self.wells:
                posterior=self.well_joint(model,wi);loc=self.local[self.row_ids[wi]]
                for start in range(0,len(loc),1024):
                    end=min(start+1024,len(loc))
                    for j in range(3):
                        pred[loc[start:end],j]=self.actions[j][(posterior[start:end,j]@self.utilities[j]).argmax(-1).cpu().numpy()]
            assert np.isfinite(pred).all()
            return pred
        finally:torch.use_deterministic_algorithms(previous)

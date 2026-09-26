"""Supervised window-consistent curve dropout; no reconstruction or target changes."""
import json
import torch
from multiscale_sequence import Network as Baseline
from masked_curve_pretrain import curve_assignment


def mask_features(x, chosen, assignments):
    # chosen=-1 is a clean window. Hide cross aliases only for corrupted windows.
    hide=(assignments[None]==chosen[:,None]) | (assignments[None]==-1)
    hide &= chosen[:,None]>=0
    hide[:,assignments==-2]=False
    return x.masked_fill(hide[:,None],0)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.register_buffer('augmentation_assignment',torch.tensor(curve_assignment()[0]))
    def forward(self,x,relative,valid,auxiliary=False):
        if self.training:
            chosen=torch.randint(9,(x.shape[0],),device=x.device)
            chosen=torch.where(torch.rand(x.shape[0],device=x.device)<.25,chosen,-1)
            x=mask_features(x,chosen,self.augmentation_assignment)
        return super().forward(x,relative,valid,auxiliary)


def self_test():
    torch.set_num_threads(2)
    a=torch.tensor(curve_assignment()[0]);x=torch.randn(2,256,206)
    y=mask_features(x,torch.tensor([0,-1]),a)
    assert torch.equal(y[1],x[1])
    assert (y[0,:,(a==0)|(a==-1)]==0).all()
    assert torch.equal(y[:,:,a==-2],x[:,:,a==-2])
    torch.manual_seed(173);base=Baseline().eval()
    torch.manual_seed(173);model=Network().eval()
    valid=torch.ones(2,256,dtype=torch.bool);depth=torch.zeros(2,256)
    with torch.no_grad():
        assert all(torch.equal(u,v) for u,v in zip(base(x,depth,valid),model(x,depth,valid)))
    print(json.dumps({'passed':True,'clean_window_unchanged':True,'depth_preserved':True,'all_depth_positions_same_mask':True,'eval_exact_baseline':True}))

if __name__=='__main__':self_test()

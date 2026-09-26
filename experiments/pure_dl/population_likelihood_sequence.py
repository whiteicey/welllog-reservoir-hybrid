import math
import torch
from torch.nn import functional as F
from depth_norm_joint_sequence import Network

def head_loss(output, target_z, target_default, supervised):
    gate, raw = output
    length = target_z.shape[1]
    if gate.shape[1] != length:
        repeat = length//gate.shape[1]
        assert repeat*gate.shape[1] == length
        gate = gate.repeat_interleave(repeat, dim=1)
        raw = raw.repeat_interleave(repeat, dim=1)
    assert supervised.shape == target_z.shape[:2] and bool(supervised.any())
    # Select first: invalid/out-of-well labels have exactly no loss or gradients.
    gate = gate[supervised].float(); raw = raw[supervised].float()
    truth = target_z[supervised]; default = target_default[supervised].float()
    weight = raw[:, :, 0].log_softmax(-1)
    mu = raw[:, :, 1]; sd = F.softplus(raw[:, :, 2])+.025
    nll = -(weight-.5*((truth[:, :, None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    return F.binary_cross_entropy_with_logits(gate, default)+nll.masked_fill(default.bool(),0).mean()



def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    main, mid, coarse = outputs
    return (head_loss(main, target_z, target_default, supervised)
            +.2*head_loss(mid, target_z, target_default, supervised)
            +.1*head_loss(coarse, target_z, target_default, supervised))



if __name__=='__main__':
    torch.manual_seed(229);torch.set_num_threads(2)
    g=torch.randn(2,8,3,requires_grad=True);r=torch.randn(2,8,3,3,8,requires_grad=True)
    y=torch.randn(2,8,3);d=torch.rand(2,8,3)>.3;m=torch.ones(2,8,dtype=torch.bool);m[:,0]=False
    result=head_loss((g,r),y,d,m)
    rr=r[m];gg=g[m];dd=d[m];yy=y[m]
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    expected=(F.binary_cross_entropy_with_logits(gg,dd.float(),reduction='none')-dist.log_prob(yy).masked_fill(dd,0)).mean()
    assert torch.allclose(result,expected,atol=1e-6)
    changed=y.clone();changed[d|(~m[:,:,None])]=1000
    assert torch.equal(result,head_loss((g,r),changed,d,m))
    result.backward();assert torch.isfinite(r.grad).all() and (r.grad[d]==0).all() and (r.grad[~m]==0).all()
    all_d=torch.ones_like(d)
    assert torch.allclose(head_loss((g,r),y,all_d,m),F.binary_cross_entropy_with_logits(g[m],all_d[m].float()))
    print('PASS: population mixture likelihood matches independent distribution; default and padded density gradients zero; all-default case BCE only')

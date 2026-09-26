import math
import torch
from torch.nn import functional as F
from blur_no_dropout_sequence import Network

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
    per_target=F.binary_cross_entropy_with_logits(gate,default,reduction='none')+nll.masked_fill(default.bool(),0)
    return (per_target*per_target.new_tensor([.3,.35,.35])).sum(-1).mean()



def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    main, mid, coarse = outputs
    return (head_loss(main, target_z, target_default, supervised)
            +.2*head_loss(mid, target_z, target_default, supervised)
            +.1*head_loss(coarse, target_z, target_default, supervised))




if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(56)
    r=torch.randn(2,16,3,3,8,requires_grad=True);g=torch.randn(2,16,3,requires_grad=True);y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.5;m=torch.ones(2,16,dtype=torch.bool);m[:,0]=False
    result=head_loss((g,r),y,d,m);rr=r[m]
    distribution=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    terms=F.binary_cross_entropy_with_logits(g[m],d[m].float(),reduction='none')-distribution.log_prob(y[m]).masked_fill(d[m],0)
    expected=sum(w*terms[:,j].mean() for j,w in enumerate([.3,.35,.35]))
    assert torch.allclose(result,expected,atol=1e-6)
    altered=y.clone();altered[d]=10000;altered[:,0]=float('nan');assert torch.equal(result,head_loss((g,r),altered,d,m))
    result.backward();assert torch.isfinite(r.grad).all() and torch.equal(r.grad[d],torch.zeros_like(r.grad[d]))
    print('PASS: independent per-target weighted distribution likelihood and masked gradients')

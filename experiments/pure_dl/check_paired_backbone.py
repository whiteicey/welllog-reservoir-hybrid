"""Check equal full-posterior mixing, independent members and bounded GPU use."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import expit, logsumexp
from scipy.stats import norm, t
from paired_backbone_sequence import Network, merge_predictions, objective
from global_cosine_sequence import Network as Single
from por_t3_patch_loss import probability

torch.set_num_threads(2); torch.manual_seed(127)
outputs = [(torch.randn(11,3)*3, torch.randn(11,3,3,8)) for _ in range(2)]
gate, raw = merge_predictions(outputs)
edges = np.r_[-np.inf,np.linspace(-4,4,30),np.inf]
assert raw.shape == (11,3,3,16)
expected_gate = sum(expit(g.numpy()) for g,_ in outputs)/2
np.testing.assert_allclose(gate.sigmoid().numpy(),expected_gate,rtol=2e-6,atol=2e-7)
max_error = 0.
for j in range(3):
    expected = np.zeros((11,len(edges)-1))
    for g,r in outputs:
        rr = r.double().numpy()
        weight = np.exp(rr[:,j,0]-logsumexp(rr[:,j,0],axis=-1,keepdims=True))
        sd = np.logaddexp(0,rr[:,j,2])+.025
        z = (edges[None,:,None]-rr[:,j,1,None,:])/sd[:,None,:]
        cdf = t.cdf(z*np.sqrt(3),df=3) if j==0 else norm.cdf(z)
        cp = (np.diff(cdf,axis=1)*weight[:,None,:]).sum(-1)
        expected += (1-expit(g.numpy()[:,j,None]))*cp/2
    cp = probability(j,raw[:,j,0].softmax(-1),raw[:,j,1],F.softplus(raw[:,j,2])+.025,
                     torch.tensor(edges,dtype=torch.float32)).numpy()
    actual = (1-gate.sigmoid().numpy()[:,j,None])*cp
    error = float(np.max(abs(actual-expected))); max_error=max(max_error,error)
    assert error < 2e-6,error
    np.testing.assert_allclose(actual.sum(-1)+expected_gate[:,j],1,atol=2e-6)
for g in [-10000.,10000.]:
    eg,er=merge_predictions([(torch.full((3,3),g),torch.randn(3,3,3,8)),
                            (torch.full((3,3),-g),torch.randn(3,3,3,8))])
    assert torch.isfinite(eg).all() and torch.isfinite(er).all()
    torch.testing.assert_close(eg.sigmoid(),torch.full_like(eg,.5))

torch.manual_seed(51); single=Single(features=12)
torch.manual_seed(51); model=Network(features=12).eval()
assert all(torch.equal(v,model.members[0].state_dict()[k]) for k,v in single.state_dict().items())
assert not torch.equal(model.members[0].stem.weight,model.members[1].stem.weight)
x=torch.randn(2,256,12);d=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:17]=False
with torch.no_grad():
    base=model(x,d,valid,absolute_depth=d)
    xx=x.clone();xx[~valid]+=100
    for a,b in zip(base,model(xx,d,valid,absolute_depth=d)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    xx=x.clone();xx[1]+=100
    for a,b in zip(base,model(xx,d,valid,absolute_depth=d)):torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
q=torch.rand(2,256,requires_grad=True)
loss=objective(model(x,d,valid,True,absolute_depth=d),torch.randn(2,256,3),
               torch.rand(2,256,3)>.5,valid,True,q,epoch=0)
loss.backward();assert q.grad is None
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
assert all(m.main_head.density.weight.grad.abs().sum()>0 for m in model.members)
result={'passed':True,'independent_scipy_full_mixture_max_error':max_error,
        'scope':'Fixed equal full-posterior mixture, member independence, first member initialization, well/padding isolation, detached Q and all-member gradients',
        'gpu_tested':False,'parameters':sum(v.numel() for v in Network().parameters())}
from gpu_job_guard_v2 import check_slot
print('GPU_ADMISSION',check_slot(),flush=True)
model=Network().cuda();torch.cuda.reset_peak_memory_stats()
x=torch.randn(64,256,206,device='cuda');d=torch.randn(64,256,device='cuda');valid=torch.ones(64,256,dtype=torch.bool,device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,d,valid,True,absolute_depth=d)
loss=objective(out,torch.randn(64,256,3,device='cuda'),torch.rand(64,256,3,device='cuda')>.5,
               valid,True,torch.ones_like(d),epoch=0)
loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
result.update(gpu_tested=True,synthetic_gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
path=Path('outputs/pure_dl_paired_backbone_sequence/preflight.json')
path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)

"""Independent direct full-density likelihood check; GPU only with --cuda."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from paired_predictive_loss import objective
from paired_backbone_sequence import Network, objective as individual_objective

p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');args=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(509)
outputs=[tuple((torch.randn(2,n,3,requires_grad=True),torch.randn(2,n,3,3,8,requires_grad=True))
               for n in [16,8,4]) for _ in range(2)]
y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.5;mask=torch.rand(2,16)>.15
q=torch.rand(2,16,requires_grad=True)

def direct_head(index):
    logs=[]
    for member in outputs:
        g,r=member[index];repeat=16//g.shape[1]
        g=g.repeat_interleave(repeat,1)[mask];r=r.repeat_interleave(repeat,1)[mask]
        target=y[mask];mu=r[:,:,1];sd=F.softplus(r[:,:,2])+.025
        densities=torch.stack([torch.distributions.StudentT(3.,mu[:,0],sd[:,0]/math.sqrt(3)).log_prob(target[:,0,None]),
                               torch.distributions.Normal(mu[:,1],sd[:,1]).log_prob(target[:,1,None]),
                               torch.distributions.Normal(mu[:,2],sd[:,2]).log_prob(target[:,2,None])],1)
        conditional=(r[:,:,0].log_softmax(-1)+densities).logsumexp(-1)
        logs.append(torch.where(d[mask],F.logsigmoid(g),F.logsigmoid(-g)+conditional))
    direct=(torch.stack(logs).logsumexp(0)-math.log(2)).mean(-1)
    return -(direct*q[mask].detach()).mean()

actual=objective(outputs,y,d,mask,True,q,epoch=0)
expected=direct_head(0)+.2*direct_head(1)+.1*direct_head(2)
torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
tensors=[t for member in outputs for head in member for t in head]
ag=torch.autograd.grad(actual,tensors,retain_graph=True)
eg=torch.autograd.grad(expected,tensors,retain_graph=True)
for a,b in zip(ag,eg):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
upper=individual_objective(outputs,y,d,mask,True,q,epoch=0)
assert actual<=upper+2e-6,'Mixture NLL must obey Jensen inequality'
half=objective(outputs,y,d,mask,True,q*.5,epoch=0)
torch.testing.assert_close(half,actual*.5)
torch.testing.assert_close(objective(outputs,y,d,mask,True,q,epoch=40),direct_head(0),atol=2e-6,rtol=2e-6)
actual.backward();assert q.grad is None
assert all(torch.isfinite(t.grad).all() for t in tensors)
all_default=torch.ones_like(d)
for t in tensors:t.grad=None
objective(outputs,y,all_default,mask,True,q,epoch=0).backward()
assert all(head[1].grad.abs().sum()==0 for member in outputs for head in member)
result={'passed':True,'direct_full_density_loss_error':float(abs(actual.detach()-expected.detach())),
        'checks':['independent Torch StudentT/Normal full-density likelihood','all-head/all-member gradients match','Jensen inequality','final Q scaling and detachment','auxiliary retirement','default continuous gradients zero'],
        'gpu_tested':False}
if args.cuda:
    from gpu_job_guard_v2 import check_slot
    print('GPU_ADMISSION',check_slot(),flush=True)
    model=Network().cuda();torch.cuda.reset_peak_memory_stats()
    x=torch.randn(64,256,206,device='cuda');depth=torch.randn(64,256,device='cuda');valid=torch.ones(64,256,dtype=torch.bool,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,depth,valid,True,absolute_depth=depth)
    loss=objective(out,torch.randn(64,256,3,device='cuda'),torch.rand(64,256,3,device='cuda')>.5,valid,True,torch.ones_like(depth),epoch=0)
    loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    result.update(gpu_tested=True,parameters=sum(v.numel() for v in model.parameters()),synthetic_gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
out=Path('outputs/pure_dl_paired_predictive_sequence');out.mkdir(exist_ok=True)
(out/('preflight.json' if args.cuda else 'preflight_cpu.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

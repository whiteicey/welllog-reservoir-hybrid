import json,time,math
from pathlib import Path
import torch
import numpy as np
from gen_flow_spline import *
OUT=Path('outputs/pure_dl_gen_spline');OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2);torch.manual_seed(109)
r=(torch.randn(200,P,dtype=torch.float64)*.3).requires_grad_();v=torch.linspace(-9,9,200,dtype=torch.float64).requires_grad_()
z,ld=transform(v,r);back,invld=transform(z,r,True)
assert (back-v).abs().max()<1e-9 and (invld+ld).abs().max()<1e-9
grad=torch.autograd.grad(z.sum(),v,retain_graph=True)[0]
assert (grad>0).all() and (grad.log()-ld).abs().max()<1e-9
# density integrates to one for varied conditional flows
rr=r[:5].detach();grid=torch.linspace(-30,30,60001,dtype=torch.float64).expand(5,-1)
zz,ll=transform(grid,rr);integral=torch.trapezoid(torch.exp(-.5*zz.square()-.5*math.log(2*math.pi)+ll),grid,dim=-1)
assert (integral-1).abs().max()<1e-5
m=Network(features=12);x=torch.randn(2,256,12);d=torch.randn(2,256);mask=torch.ones(2,256,dtype=torch.bool);mask[:,0]=False
y=torch.randn(2,256,3);default=torch.rand(2,256,3)>.4
outputs=m(x,d,mask,True,absolute_depth=d);loss=objective(outputs,y,default,mask,True)
y2=y.clone();y2[default|(~mask[:,:,None])]=1e6
assert torch.equal(loss,objective(outputs,y2,default,mask,True))
loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
report={'inverse_max_error':float((back-v).abs().max()),'log_jacobian_autograd_error':float((grad.log()-ld).abs().max()),'integral_max_error':float((integral-1).abs().max()),'masked_labels_ignored':True,'finite_model_gradients':True}
(OUT/'numerical_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__' and '--gpu' in __import__('sys').argv:
    from gpu_job_guard import check_slot
    print(check_slot());m=Network().cuda();x=torch.randn(64,256,206,device='cuda');d=torch.randn(64,256,device='cuda');mask=torch.ones_like(d,dtype=torch.bool);y=torch.randn(64,256,3,device='cuda');default=torch.rand_like(y)>.4
    opt=torch.optim.AdamW(m.parameters(),lr=.001);torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(8):
        torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=m(x,d,mask,True,absolute_depth=d)
        loss=objective(outputs,y,default,mask,True);assert torch.isfinite(loss);loss.backward();opt.step();torch.cuda.synchronize()
        if i>=3:times.append(time.perf_counter()-t)
    report.update(parameters=sum(p.numel() for p in m.parameters()),peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,mean_update_seconds=float(np.mean(times)),warmup_steps=3,measured_steps=5)
    (OUT/'benchmark.json').write_text(json.dumps(report,indent=2));print(report)

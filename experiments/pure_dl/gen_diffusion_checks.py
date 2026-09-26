import json,time
from pathlib import Path
import torch,numpy as np
from gen_diffusion_model import *
out=Path('outputs/pure_dl_gen_diffusion');out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(119)
m=Network(features=12);x=torch.randn(2,256,12);depth=torch.randn(2,256);mask=torch.ones(2,256,dtype=torch.bool);mask[:,0]=False
y=torch.randn(2,256,3);d=torch.rand(2,256,3)>.4;o=m(x,depth,mask,True,absolute_depth=depth)
e=torch.randn(int(mask.sum()),3);t=torch.rand_like(e);changed=y.clone();changed[d|~mask[:,:,None]]=1e6
assert torch.equal(m.head_loss(o[0],y,d,mask,e,t),m.head_loss(o[0],changed,d,mask,e,t))
loss=m.objective(o,y,d,mask,True);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
c=torch.randn(4,3,CONTEXT);a=m.generate(c);b=m.generate(c);assert torch.equal(a,b)
p=probabilities(a[:,0],torch.linspace(-4,4,257));assert torch.equal(p.sum(-1),torch.ones(4))
# Independent orthogonal diffusion rotation inversion for all t including endpoints.
yy=torch.randn(100,dtype=torch.float64);ee=torch.randn(100,dtype=torch.float64);tt=torch.linspace(0,1,100,dtype=torch.float64)
aa=torch.cos(torch.pi*tt/2);ss=torch.sin(torch.pi*tt/2);zz=aa*yy+ss*ee;vv=aa*ee-ss*yy
error=max(float((aa*zz-ss*vv-yy).abs().max()),float((ss*zz+aa*vv-ee).abs().max()))
assert error<1e-12
report={'masked_labels_ignored':True,'finite_all_gradients':True,'deterministic_quantile_generation':True,'histogram_mass_exact':True,'rotation_inversion_error':error,'samples':SAMPLES,'steps':STEPS}
(out/'numerical_checks.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if '--gpu' in __import__('sys').argv:
    from gpu_job_guard_v2 import check_slot
    print(check_slot(),flush=True);m=Network().cuda();x=torch.randn(64,256,206,device='cuda');depth=torch.randn(64,256,device='cuda');mask=torch.ones_like(depth,dtype=torch.bool);mask[:,:64]=False;mask[:,192:]=False;y=torch.randn(64,256,3,device='cuda');d=torch.rand_like(y)>.5;opt=torch.optim.AdamW(m.parameters(),lr=.001)
    torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(8):
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):o=m(x,depth,torch.ones_like(mask),True,absolute_depth=depth)
        loss=m.objective(o,y,d,mask,True);assert torch.isfinite(loss);loss.backward();opt.step();torch.cuda.synchronize()
        if i>=3:times.append(time.perf_counter()-start)
    ctx=torch.randn(4096,3,CONTEXT,device='cuda');evaltimes=[]
    for i in range(4):
        torch.cuda.synchronize();start=time.perf_counter();samples=m.generate(ctx)
        assert torch.isfinite(samples).all()
        for j in range(3):p=probabilities(samples[:,j],torch.linspace(-4,4,257,device='cuda'));assert torch.all(p.sum(-1)==1)
        torch.cuda.synchronize()
        if i:evaltimes.append(time.perf_counter()-start)
    report.update(parameters=sum(p.numel() for p in m.parameters()),training_update_seconds=float(np.mean(times)),generation_4096rows_seconds=float(np.mean(evaltimes)),projected_55epoch_seconds=55*(64*np.mean(times)+36*np.mean(evaltimes)),peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,warmup_training_steps=3,measured_training_steps=5)
    (out/'benchmark.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

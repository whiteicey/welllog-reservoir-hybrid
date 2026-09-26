import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t,norm
from por_t3_patch_loss import objective,components,point_statistics
from por_student3_sequence import self_test
from patch_hierarchy_sequence import Network
torch.set_num_threads(3);torch.manual_seed(345)
old_check=self_test()
gate=torch.randn(2,16,3,requires_grad=True);raw=torch.randn(2,16,3,3,8,requires_grad=True)
y=torch.randn(2,16,3);default=torch.rand(2,16,3)>.7;mask=torch.ones(2,16,dtype=torch.bool);mask[:,0]=False
q=torch.rand(2,16,requires_grad=True)
r=raw[mask].float();target=y[mask];d=default[mask];sd=torch.nn.functional.softplus(r[:,:,2])+.025
distributions=[torch.distributions.StudentT(3,r[:,0,1],sd[:,0]/math.sqrt(3)),torch.distributions.Normal(r[:,1,1],sd[:,1]),torch.distributions.Normal(r[:,2,1],sd[:,2])]
nll=torch.stack([-torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=r[:,j,0]),distributions[j]).log_prob(target[:,j]) for j in range(3)],1)
point=(torch.nn.functional.binary_cross_entropy_with_logits(gate[mask],d.float(),reduction='none')+nll.masked_fill(d,0)).mean(-1)
expected=(q[mask].detach()*point).mean()
loss=objective((gate,raw),y,default,mask,weights=q,epoch=0)
assert torch.allclose(loss,expected,atol=1e-6)
loss.backward();assert q.grad is None and (raw.grad[default]==0).all() and (raw.grad[~mask]==0).all()
edges=np.r_[-np.inf,np.linspace(-5,5,255),np.inf];edges=[edges]*3
with torch.no_grad():stats,cdf=point_statistics(gate[mask],r,target,d,edges)
assert torch.allclose(stats['nll'][~d],nll[~d],atol=1e-6)
mu=r[:,:,1].detach().numpy();s=sd.detach().numpy();w=r[:,:,0].detach().softmax(-1).numpy();yy=target.numpy()
independent=np.stack([(w[:,j]*(t.cdf((yy[:,j,None]-mu[:,j])/(s[:,j]/np.sqrt(3)),3) if j==0 else norm.cdf((yy[:,j,None]-mu[:,j])/s[:,j]))).sum(-1) for j in range(3)],1)
tail=2*np.minimum(independent,1-independent);error=float(np.max(abs(tail[~d.numpy()]-stats['target_tail'].numpy()[~d.numpy()])))
assert error<5e-7 and np.isnan(stats['target_tail'].numpy()[d.numpy()]).all()
assert torch.isfinite(cdf).all() and torch.allclose(cdf[:,:,-1],torch.ones_like(cdf[:,:,-1]),atol=1e-6)
model=Network().cuda();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');depth=torch.zeros(64,256,device='cuda');target=torch.zeros(64,256,3,device='cuda');d=torch.zeros_like(target,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(x,depth,v,True,absolute_depth=depth)
loss=objective(outputs,target,d,mask,True,torch.ones_like(depth),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
result={'passed':True,'old_independent_probability_check':old_check,'independent_weighted_nll':True,'probe_tail_error':error,'default_mask_and_detached_Q':True,'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'parameters':sum(p.numel() for p in model.parameters())}
out=Path('outputs/pure_dl_por_t3_patch_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

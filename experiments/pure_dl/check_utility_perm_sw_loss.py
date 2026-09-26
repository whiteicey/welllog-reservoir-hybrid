"""Independent scalar quadrature, loss masking/weighting and real GPU backward."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr, softmax
from utility_perm_sw_loss import expected_utility, objective
from patch_hierarchy_sequence import Network

torch.set_num_threads(3)
torch.manual_seed(910)
med, scale = [2., 0., 4.], [1., 1.5, .3]
r = torch.randn(5, 3, 3, 8, dtype=torch.float64)*.3
y = torch.randn(5, 3, dtype=torch.float64)*.2
actual = expected_utility(r, y, med, scale).numpy()
reference = np.zeros_like(actual)
for i in range(5):
    for j in range(3):
        z = float(y[i,j])*scale[j]+med[j]
        truth = [np.expm1, lambda x:10**x, np.exp][j](z)
        means = r[i,j,1].numpy()
        std = np.logaddexp(0,r[i,j,2].numpy())+.025
        weights = softmax(r[i,j,0].numpy())
        def integral(t):
            if j == 1: low, high = z-t, z+t
            else:
                radius = [.08,0,.05][j]*(abs(truth)+1e-5)
                transform = np.log1p if j == 0 else np.log
                low, high = transform(truth-t*radius), transform(truth+t*radius)
            a = ((low-med[j])/scale[j]-means)/std
            b = ((high-med[j])/scale[j]-means)/std
            return np.dot(weights,ndtr(b)-ndtr(a))
        reference[i,j] = quad(integral,0,1,epsabs=1e-11)[0]
error = float(np.max(np.abs(reference-actual)))
assert error < 1e-6, error
gate = torch.randn(2,8,3,requires_grad=True)
raw = torch.randn(2,8,3,3,8,requires_grad=True)
target = torch.randn(2,8,3)*.1
default = torch.zeros(2,8,3,dtype=torch.bool);default[:,0] = True
mask = torch.ones(2,8,dtype=torch.bool);mask[:,-1] = False
weight = torch.ones(2,8,requires_grad=True)
args = dict(med=med,scale=scale)
a = objective((gate,raw),target,default,mask,weights=weight,**args)
b = objective((gate,raw),target,default,mask,weights=weight*.2,**args)
assert torch.allclose(b,a*.2)
from global_point_loss import objective as original_objective
original = original_objective((gate,raw),target,default,mask,weights=weight)
extra_gradient = torch.autograd.grad(a-original,raw,retain_graph=True)[0]
assert extra_gradient[:,:,0].abs().max()<1e-7
assert extra_gradient[:,:,1:].abs().sum()>0
a.backward()
assert weight.grad is None
assert raw.grad[:,0].abs().max()==0 and raw.grad[:,-1].abs().max()==0
assert gate.grad[:,0].abs().sum()>0
assert torch.isfinite(raw.grad).all()
model = Network().cuda()
x = torch.randn(64,256,206,device='cuda')
valid = torch.ones(64,256,dtype=torch.bool,device='cuda')
relative = torch.zeros(64,256,device='cuda')
target = torch.zeros(64,256,3,device='cuda')
default = torch.zeros_like(target,dtype=torch.bool)
mask = valid.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):
    outputs=model(x,relative,valid,auxiliary=True,absolute_depth=relative)
loss=objective(outputs,target,default,mask,True,weights=torch.ones_like(relative),**args)
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
result={'passed':True,'scalar_quadrature_max_error':error,'parameters':sum(p.numel() for p in model.parameters()),
        'gpu_synthetic_peak_mib':torch.cuda.max_memory_allocated()/1024**2,
        'default_continuous_gradient_zero':True,'point_weights_detached_and_final':True}
out=Path('outputs/pure_dl_utility_perm_sw_sequence');out.mkdir(exist_ok=True)
(out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

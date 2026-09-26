import json
from pathlib import Path
import torch
from patch_dispersion_sequence import Network,masked_dispersion
from global_cosine_sequence import Network as Original
from patch_hierarchy_sequence import SmallAttention as DotAttention
from cosine_patch_sequence import SmallAttention as CosineAttention
from por_t3_patch_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval();torch.manual_seed(89);b=Network().eval()
assert type(b.stem.hierarchy.local) is DotAttention and type(b.stem.hierarchy.global_patches) is CosineAttention
assert all(torch.equal(value,b.state_dict()[key]) for key,value in a.state_dict().items())
from fold5_screening_800 import self_test
self_test()
values=torch.randn(4,16,32,requires_grad=True);m=torch.ones(4,16,dtype=torch.bool);m[0]=False;m[1,1:]=False;m[2,8:]=False
with torch.no_grad():
    actual=masked_dispersion(values,m)
    for i in [1,2,3]:
        expected=(values[i,m[i]].var(0,unbiased=False)+1e-6).sqrt()-.001
        assert torch.allclose(actual[i],expected,atol=1e-6)
    assert (actual[0]==0).all() and (actual[1].abs()<1e-6).all()
masked_dispersion(values,m).sum().backward();assert torch.isfinite(values.grad).all()
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    assert all(torch.equal(q,w) for q,w in zip(a(x,d,v,absolute_depth=d),b(x,d,v,absolute_depth=d)))
    torch.nn.init.normal_(b.stem.hierarchy.output.weight,std=.02)
    result=b(x,d,v,absolute_depth=d);changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    assert (b.stem.hierarchy(torch.randn(2,64,256),torch.zeros_like(v))==0).all()
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');defaults=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,defaults,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
assert b.stem.hierarchy.global_patches.log_scale.grad.abs().sum()>0
assert b.stem.hierarchy.dispersion.weight.grad.abs().sum()>0
r={'passed':True,'mean_plus_dispersion':True,'empty_singleton_finite_gradients':True,'independent_variance':True,'initial_identity':True,'well_padding_empty_isolation':True,'finite_GPU_temperature_gradient':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
out=Path('outputs/pure_dl_patch_dispersion_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

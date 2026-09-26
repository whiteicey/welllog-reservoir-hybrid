import json,math
from pathlib import Path
import torch
from cosine_patch_sequence import Network,cosine_scores
from patch_hierarchy_sequence import Network as Original
from por_t3_patch_loss import objective
torch.set_num_threads(3);torch.manual_seed(17)
q=torch.randn(2,4,9,8);k=torch.randn(2,4,9,8);scale=torch.full((4,),math.log(math.sqrt(8)))
actual=cosine_scores(q,k,scale)
expected=(q@k.transpose(-1,-2))/(q.norm(dim=-1)[...,None]*k.norm(dim=-1)[...,None,:])*math.sqrt(8)
assert torch.allclose(actual,expected,atol=1e-6)
assert torch.allclose(actual,cosine_scores(q*3,k*.2,scale),atol=1e-6)
assert (cosine_scores(torch.zeros_like(q),torch.zeros_like(k),scale)==0).all()
torch.manual_seed(89);a=Original().eval();torch.manual_seed(89);b=Network().eval()
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
assert b.stem.hierarchy.local.log_scale.grad.abs().sum()>0 and b.stem.hierarchy.global_patches.log_scale.grad.abs().sum()>0
r={'passed':True,'independent_cosine_formula_and_amplitude_invariance':True,'initial_identity':True,'padding_well_empty_isolation':True,'finite_GPU_and_nonzero_scale_gradients':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
out=Path('outputs/pure_dl_cosine_patch_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

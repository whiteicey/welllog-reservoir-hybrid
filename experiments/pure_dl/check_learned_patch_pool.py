import json
from pathlib import Path
import torch
from learned_patch_pool_sequence import Network
from patch_hierarchy_sequence import Network as Original
from annealed_aux_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval()
torch.manual_seed(89);b=Network().eval()
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    assert all(torch.equal(q,w) for q,w in zip(a(x,d,v,absolute_depth=d),b(x,d,v,absolute_depth=d)))
    torch.nn.init.normal_(a.stem.hierarchy.output.weight,std=.02)
    b.stem.hierarchy.output.weight.copy_(a.stem.hierarchy.output.weight)
    assert all(torch.allclose(q,w,atol=2e-6,rtol=2e-6) for q,w in zip(a(x,d,v,absolute_depth=d),b(x,d,v,absolute_depth=d)))
    torch.nn.init.normal_(b.stem.hierarchy.pool_score.weight,std=.1)
    result=b(x,d,v,absolute_depth=d)
    changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    h=b.stem.hierarchy(torch.randn(2,64,256),torch.zeros_like(v))
    assert torch.isfinite(h).all() and (h==0).all()
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');state=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,state,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
assert b.stem.hierarchy.pool_score.weight.grad.abs().sum()>0
r={'passed':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'initial_identity':True,'learned_pool_well_padding_isolation':True,'pool_nonzero_gradient':True}
out=Path('outputs/pure_dl_learned_patch_pool_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

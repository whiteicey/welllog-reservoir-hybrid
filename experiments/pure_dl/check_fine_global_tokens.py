import json
from pathlib import Path
import torch
from fine_global_tokens_sequence import Network
from patch_hierarchy_sequence import Network as Original
from annealed_aux_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval()
torch.manual_seed(89);b=Network().eval()
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    assert all(torch.equal(q,w) for q,w in zip(a(x,d,v,absolute_depth=d),b(x,d,v,absolute_depth=d)))
    torch.nn.init.normal_(b.stem.hierarchy.output.weight,std=.02)
    result=b(x,d,v,absolute_depth=d)
    changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    assert (b.stem.hierarchy(torch.randn(2,64,256),torch.zeros_like(v))==0).all()
    seen=[]
    handle=b.stem.hierarchy.global_patches.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
    b(x,d,v,absolute_depth=d);handle.remove();assert seen==[torch.Size([2,32,32])]
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');state=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,state,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
r={'passed':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'initial_identity':True,'padding_well_isolation':True,'global_token_shape':[2,32,32]}
out=Path('outputs/pure_dl_fine_global_tokens_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

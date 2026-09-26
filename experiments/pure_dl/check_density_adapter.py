import json
from pathlib import Path
import torch
from density_adapter_sequence import Network
from patch_hierarchy_sequence import Network as Original
from annealed_aux_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval()
torch.manual_seed(89);b=Network().eval()
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    original=a(x,d,v,absolute_depth=d);initial=b(x,d,v,absolute_depth=d)
    assert all(torch.equal(q,w) for q,w in zip(original,initial))
    torch.nn.init.normal_(b.main_head.output.weight,std=.02)
    result=b(x,d,v,absolute_depth=d)
    assert torch.equal(result[0],initial[0]) and not torch.equal(result[1],initial[1])
    changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
z=torch.randn(2,128,16,requires_grad=True);g,r=b.main_head(z);g.sum().backward()
assert b.main_head.output.weight.grad is None
b.zero_grad(set_to_none=True)
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');state=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,state,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
assert b.main_head.output.weight.grad.abs().sum()>0
result={'passed':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'initial_identity':True,'adapter_does_not_directly_change_gate':True,'gate_loss_no_adapter_gradient':True,'well_padding_isolation':True}
out=Path('outputs/pure_dl_density_adapter_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

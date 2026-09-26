import json
from pathlib import Path
import torch
from dense_local_sequence import Network
from patch_hierarchy_sequence import Network as Original
from blur_ungated_sequence import UngatedBlock
from annealed_aux_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval()
torch.manual_seed(89);b=Network().eval()
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
blocks=[m.depthwise for m in b.modules() if isinstance(m,UngatedBlock)]
assert len(blocks)==9 and all(c.dilation==(1,) and c.padding==(3,) for c in blocks)
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    result=b(x,d,v,absolute_depth=d)
    assert not torch.equal(result[1],a(x,d,v,absolute_depth=d)[1])
    changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    assert all(torch.isfinite(q).all() for q in b(x,d,torch.zeros_like(v),absolute_depth=d))
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');state=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,state,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
r={'passed':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'same_initial_parameters':True,'nine_dilation1_blocks':True,'well_padding_isolation':True}
out=Path('outputs/pure_dl_dense_local_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

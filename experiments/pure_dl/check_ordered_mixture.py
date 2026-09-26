import json
from pathlib import Path
import torch
from ordered_mixture_sequence import Network
from patch_hierarchy_sequence import Network as Original
from por_t3_patch_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval();torch.manual_seed(89);b=Network().eval()
assert all(torch.equal(value,b.state_dict()[key]) for key,value in a.state_dict().items())
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    aa=a(x,d,v,True,absolute_depth=d);bb=b(x,d,v,True,absolute_depth=d)
    for (ag,ar),(bg,br) in zip(aa,bb):
        assert torch.equal(ag,bg)
        assert torch.equal(ar[:,:,:,0,:],br[:,:,:,0,:]) and torch.equal(ar[:,:,:,2,:],br[:,:,:,2,:])
        assert torch.equal(ar[:,:,:,1,:].sort(-1).values,br[:,:,:,1,:])
        assert (br[:,:,:,1,:].diff(dim=-1)>=0).all()
    result=bb[0];changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');defaults=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,defaults,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
r={'passed':True,'same_initial_parameters':True,'all_heads_means_sorted_weights_scales_unchanged':True,'padding_well_isolation':True,'finite_GPU_gradients':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
out=Path('outputs/pure_dl_ordered_mixture_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

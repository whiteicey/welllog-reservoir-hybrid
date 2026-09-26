import json
from pathlib import Path
import torch
from nearest_decoder_sequence import Network
from patch_hierarchy_sequence import Network as Original
from annealed_aux_loss import objective
torch.set_num_threads(3);torch.manual_seed(89);a=Original().eval()
torch.manual_seed(89);b=Network().eval()
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
x=torch.randn(2,256,206);d=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    result=b(x,d,v,absolute_depth=d)
    assert not torch.equal(result[1],a(x,d,v,absolute_depth=d)[1])
    changed=x.clone();changed[~v]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,v,absolute_depth=d)))
    assert all(torch.isfinite(q).all() for q in b(x,d,torch.zeros_like(v),absolute_depth=d))
    # Inspect actual inputs to decoder merges, not a duplicate standalone interpolation.
    before={};checks=[];handles=[]
    for key,module in [('merge2',b.bottleneck),('merge1',b.dec2),('merge0',b.dec1)]:
        handles.append(module.register_forward_hook(lambda mod,args,out,key=key:before.__setitem__(key,out)))
    for key in ['merge2','merge1','merge0']:
        def inspect(mod,args,key=key):
            old=before[key];upsampled=args[0][:,:old.shape[1]]
            checks.append(torch.equal(upsampled,old.repeat_interleave(2,-1)))
        handles.append(getattr(b,key).register_forward_pre_hook(inspect))
    b(x,d,v,absolute_depth=d)
    for handle in handles:handle.remove()
    assert checks==[True,True,True]
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');state=torch.zeros_like(y,dtype=torch.bool);mask=v.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,v,True,absolute_depth=d)
loss=objective(outputs,y,state,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
r={'passed':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'same_initial_parameters':True,'three_actual_merge_inputs_nearest_verified':True,'well_padding_isolation':True}
out=Path('outputs/pure_dl_nearest_decoder_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))

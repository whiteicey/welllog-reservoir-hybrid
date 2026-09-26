import json
from pathlib import Path
import torch
from ema_context_sequence import Network,normalized_filter
from patch_hierarchy_sequence import Network as Original
from por_t3_patch_loss import objective
torch.set_num_threads(3);torch.manual_seed(41)
x=torch.randn(2,3,33);mask=torch.rand(2,33)>.25;decay=torch.tensor([.2,.8,.98],requires_grad=True)
actual=normalized_filter(x,mask,decay)
num=torch.zeros(2,3);den=torch.zeros(2,3);rows=[]
for i in range(33):
    m=mask[:,i,None];num=decay*num+(1-decay)*x[:,:,i]*m;den=decay*den+(1-decay)*m
    rows.append((num/den.clamp_min(1e-6))*m)
reference=torch.stack(rows,-1);error=float((actual-reference).abs().max().detach());assert error<2e-5,error
actual.square().mean().backward();assert torch.isfinite(decay.grad).all()
torch.manual_seed(89);a=Original().eval();torch.manual_seed(89);b=Network().eval()
x=torch.randn(2,256,206);d=torch.zeros(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:19]=False
with torch.no_grad():
    assert all(torch.equal(q,w) for q,w in zip(a(x,d,valid,absolute_depth=d),b(x,d,valid,absolute_depth=d)))
    torch.nn.init.normal_(b.stem.context.output.weight,std=.02)
    result=b(x,d,valid,absolute_depth=d);changed=x.clone();changed[~valid]+=100
    assert all(torch.equal(q,w) for q,w in zip(result,b(changed,d,valid,absolute_depth=d)))
    changed=x.clone();changed[1]+=100
    assert all(torch.equal(q[0],w[0]) for q,w in zip(result,b(changed,d,valid,absolute_depth=d)))
    assert (b.stem.context(torch.randn(2,64,256),torch.zeros_like(valid))==0).all()
b=b.cuda().train();x=torch.randn(64,256,206,device='cuda');valid=torch.ones(64,256,dtype=torch.bool,device='cuda');d=torch.zeros(64,256,device='cuda')
y=torch.zeros(64,256,3,device='cuda');defaults=torch.zeros_like(y,dtype=torch.bool);mask=valid.clone();mask[:,:64]=False;mask[:,192:]=False
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=b(x,d,valid,True,absolute_depth=d)
loss=objective(outputs,y,defaults,mask,True,torch.ones_like(d),epoch=20);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
assert b.stem.context.rate.grad.abs().sum()>0
result={'passed':True,'fft_vs_independent_recurrence_max_error':error,'initial_identity':True,'well_padding_empty_isolation':True,'rate_gradient_nonzero':True,'parameters':sum(p.numel() for p in b.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
out=Path('outputs/pure_dl_ema_context_sequence');out.mkdir(exist_ok=True);(out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

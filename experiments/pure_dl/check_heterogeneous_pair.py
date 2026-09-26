"""Check changed member structure and gradients; reuse verified mixture algebra."""
import json
from pathlib import Path
import torch
from heterogeneous_pair_sequence import Network
from paired_backbone_sequence import Network as Baseline, objective
from cosine_patch_sequence import SmallAttention as CosineAttention
from patch_hierarchy_sequence import SmallAttention as DotAttention
from gpu_job_guard_v2 import check_slot

torch.set_num_threads(2)
torch.manual_seed(313); a=Baseline(features=12).eval()
torch.manual_seed(313); m=Network(features=12).eval()
assert isinstance(m.members[0].stem.hierarchy.local,DotAttention)
assert isinstance(m.members[1].stem.hierarchy.local,CosineAttention)
assert all(isinstance(x.stem.hierarchy.global_patches,CosineAttention) for x in m.members)
assert 'members.0.stem.hierarchy.local.log_scale' not in m.state_dict()
assert m.state_dict()['members.1.stem.hierarchy.local.log_scale'].shape==(4,)
x=torch.randn(2,256,12); d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
with torch.no_grad():
    original=a(x,d,v,absolute_depth=d); actual=m(x,d,v,absolute_depth=d)
    for q,w in zip(original,actual):torch.testing.assert_close(q,w,rtol=0,atol=0)
    # Activate zero-initialized branches to check actual changed attention paths.
    for member in m.members:member.stem.hierarchy.output.weight.normal_(std=.02)
    actual=m(x,d,v,absolute_depth=d)
    xx=x.clone();xx[~v]+=100
    for q,w in zip(actual,m(xx,d,v,absolute_depth=d)):torch.testing.assert_close(q,w,rtol=0,atol=0)
    xx=x.clone();xx[1]+=100
    for q,w in zip(actual,m(xx,d,v,absolute_depth=d)):torch.testing.assert_close(q[0],w[0],rtol=0,atol=0)
    empty=m(x,d,torch.zeros_like(v),absolute_depth=d)
    assert all(torch.isfinite(t).all() for t in empty)
q=torch.ones(2,256,requires_grad=True)
loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True,q,epoch=0)
loss.backward();assert q.grad is None
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
assert m.members[1].stem.hierarchy.local.log_scale.grad.abs().sum()>0
print('GPU_ADMISSION',check_slot(),flush=True)
m=Network().cuda();torch.cuda.reset_peak_memory_stats()
x=torch.randn(64,256,206,device='cuda');d=torch.randn(64,256,device='cuda');v=torch.ones(64,256,dtype=torch.bool,device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):outputs=m(x,d,v,True,absolute_depth=d)
loss=objective(outputs,torch.randn(64,256,3,device='cuda'),torch.rand(64,256,3,device='cuda')>.5,v,True,torch.ones_like(d),epoch=0)
loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
result={'passed':True,'parameters':sum(p.numel() for p in m.parameters()),'synthetic_gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20,
        'checks':['actual local/global member types','initial prediction identity vs homogeneous pair','active-branch well/padding isolation','empty mask finite','detached Q','new local temperature nonzero gradient','normalGPU bf16 finite gradients'],
        'reused_unchanged_mixture_proof':'outputs/pure_dl_paired_backbone_sequence/preflight.json','new_gate':80.2}
out=Path('outputs/pure_dl_heterogeneous_pair_sequence');out.mkdir(exist_ok=True)
(out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

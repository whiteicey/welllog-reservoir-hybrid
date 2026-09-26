"""CPU-only tests of the inspired residual attention-map contrast."""
import json,math
from pathlib import Path
import numpy as np
import torch
from contrastive_global_sequence import Network,ContrastiveAttention
from global_cosine_sequence import Network as Baseline
from por_t3_patch_loss import objective
from fold5_screening_802 import self_test

def array(x):return x.detach().cpu().numpy()
def softmax(x):
    y=np.exp(x-x.max(-1,keepdims=True));return y/y.sum(-1,keepdims=True)

def main():
    torch.set_num_threads(2);self_test()
    torch.manual_seed(38);a=Baseline().eval();rng=torch.get_rng_state()
    torch.manual_seed(38);b=Network().eval();assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in b.parameters())==663553
    x=torch.randn(2,256,206);d=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:19]=False
    with torch.no_grad():
        aa=a(x,d,valid,True,absolute_depth=d);bb=b(x,d,valid,True,absolute_depth=d)
        assert all(torch.equal(q,w) for a1,b1 in zip(aa,bb) for q,w in zip(a1,b1))
        # Alpha=0 must preserve A1 even after outer zero projection is activated.
        a.stem.hierarchy.output.weight.normal_(0,.02);b.stem.hierarchy.output.weight.copy_(a.stem.hierarchy.output.weight)
        aa=a(x,d,valid,True,absolute_depth=d);bb=b(x,d,valid,True,absolute_depth=d)
        assert all(torch.equal(q,w) for a1,b1 in zip(aa,bb) for q,w in zip(a1,b1))
        global_attention=b.stem.hierarchy.global_patches
        global_attention.contrast_alpha.copy_(torch.tensor([.3,-.4,.5,-.6]))
        active=b(x,d,valid,True,absolute_depth=d)
        assert any((q-w).abs().max()>0 for a1,b1 in zip(aa,active) for q,w in zip(a1,b1))
        changed=x.clone();changed[~valid]+=1e4
        padded=b(changed,d,valid,True,absolute_depth=d)
        assert all(torch.equal(q,w) for a1,b1 in zip(active,padded) for q,w in zip(a1,b1))
        changed=x.clone();changed[1]+=100
        other=b(changed,d,valid,True,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for a1,b1 in zip(active,other) for q,w in zip(a1,b1))
        shifted=b(x,d,valid,True,absolute_depth=d+1000)
        assert all(torch.equal(q,w) for a1,b1 in zip(active,shifted) for q,w in zip(a1,b1))
        assert (b.stem.hierarchy(torch.randn(2,64,256),torch.zeros_like(valid))==0).all()
    layer=ContrastiveAttention().eval();layer.contrast_alpha.data.copy_(torch.tensor([.3,-.4,2.,-2.]))
    layer.relative_bias.data.normal_(0,.1)
    token=torch.randn(2,16,32);mask=torch.ones(2,16,dtype=torch.bool);mask[0,:5]=False
    mixed,values,a1,a2=layer.attention_maps(token,mask)
    # NumPy independently reconstructs layernorm, projections and both normalized QK maps.
    z=array(token);z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+layer.norm.eps)
    z=z*array(layer.norm.weight)+array(layer.norm.bias);z=np.where(array(mask)[...,None],z,0)
    qkv=(z@array(layer.qkv.weight).T).reshape(2,16,3,4,8).transpose(2,0,3,1,4)
    qk2=(z@array(layer.contrast_qk.weight).T).reshape(2,16,2,4,8).transpose(2,0,3,1,4)
    pos=np.arange(16);bias=array(layer.relative_bias)[:,pos[:,None]-pos[None,:]+15]
    scale=np.exp(np.minimum(array(layer.log_scale),math.log(100.)))[None,:,None,None]
    maps=[]
    for q,k in [(qkv[0],qkv[1]),(qk2[0],qk2[1])]:
        q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-6)
        k=k/np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-6)
        score=(q@k.swapaxes(-2,-1))*scale+bias
        maps.append(softmax(np.where(array(mask)[:,None,None,:],score,-1e4)))
    coefficient=.5*np.tanh(array(layer.contrast_alpha))[None,:,None,None]
    expected=maps[0]+coefficient*(maps[0]-maps[1])
    formula_error=float(np.max(abs(array(mixed)-expected)));assert formula_error<2e-6
    row_sum_error=float((mixed.sum(-1)-1).abs().max());assert row_sum_error<5e-7
    assert torch.equal(mixed[0,:,:,:5],torch.zeros_like(mixed[0,:,:,:5]))
    constant=torch.randn(2,4,1,8).expand(2,4,16,8)
    difference=((a1-a2)@constant).abs().max()
    assert difference<2e-6 and torch.allclose(mixed@constant,a1@constant,atol=2e-6)
    expected_output=(expected@qkv[2]).transpose(0,2,1,3).reshape(2,16,32)@array(layer.project.weight).T
    expected_output=np.where(array(mask)[...,None],expected_output,0)
    output_error=float(np.max(abs(array(layer(token,mask))-expected_output)));assert output_error<2e-6
    assert torch.equal(layer(token,torch.zeros_like(mask)),torch.zeros_like(token))
    # Loss/Q contract and gradients in the fully activated new path.
    b.train();outputs=b(x,d,valid,True,absolute_depth=d)
    target=torch.randn(2,256,3);default=torch.rand(2,256,3)>.5
    use=valid.clone();use[:,:64]=False;use[:,192:]=False
    q=torch.ones(2,256,requires_grad=True)
    loss=objective(outputs,target,default,use,True,q,epoch=20)
    assert torch.allclose(objective(outputs,target,default,use,True,q*.2,epoch=20),loss*.2,atol=1e-7)
    loss.backward();assert q.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
    assert b.stem.hierarchy.global_patches.contrast_alpha.grad.abs().sum()>0
    assert b.stem.hierarchy.global_patches.contrast_qk.weight.grad.abs().sum()>0
    assert all(p.device.type=='cpu' for p in b.parameters())
    summary={'passed':True,'scope':'CPU only; no GPU or training','parameters':663553,'added_parameters':2052,'old_state_and_RNG_identical':True,'initial_full_output_identity':True,'alpha0_identity_with_active_outer_projection':True,'independent_numpy_attention_error':formula_error,'independent_numpy_output_error':output_error,'attention_row_sum_error':row_sum_error,'constant_value_difference_max':float(difference.detach()),'active_padding_well_empty_mask_isolation':True,'absolute_depth_invariance':True,'new_alpha_and_QK_gradients':True,'Q_detached_final_loss_scaling':True,'strict_threshold':80.2}
    out=Path('outputs/pure_dl_contrastive_global_sequence');out.mkdir(exist_ok=True)
    (out/'cpu_preflight.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

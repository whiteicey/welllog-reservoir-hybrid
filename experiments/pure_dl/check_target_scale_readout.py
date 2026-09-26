"""CPU-only independent routing, isolation, and loss-contract checks."""
import json
from pathlib import Path
import numpy as np
import torch
from target_scale_readout_sequence import Network,TargetScaleReadout
from global_cosine_sequence import Network as Baseline
from por_t3_patch_loss import objective
from fold5_screening_802 import self_test

def array(x):return x.detach().cpu().numpy()

def interpolate_numpy(x,length):
    pos=np.clip((np.arange(length)+.5)*x.shape[-1]/length-.5,0,x.shape[-1]-1)
    lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,x.shape[-1]-1);alpha=pos-lo
    return x[...,lo]*(1-alpha)+x[...,hi]*alpha

def main():
    torch.set_num_threads(2);self_test()
    torch.manual_seed(91);a=Baseline().eval();expected_rng=torch.get_rng_state()
    torch.manual_seed(91);b=Network().eval()
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in b.parameters())==682269
    x=torch.randn(2,256,206);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
    with torch.no_grad():
        original=a(x,d,v,True,absolute_depth=d);actual=b(x,d,v,True,absolute_depth=d)
        assert all(torch.equal(q,w) for aa,bb in zip(original,actual) for q,w in zip(aa,bb))
        b.target_scale.output.weight.normal_(0,.02)
        capture={}
        hook=b.target_scale.register_forward_hook(lambda module,args,out:capture.update(inputs=args,residual=out))
        active=b(x,d,v,True,absolute_depth=d);hook.remove()
        fine=capture['inputs'][0].transpose(1,2);residual=capture['residual']
        manual=[]
        for target in range(3):
            rows=slice(target*24,(target+1)*24)
            manual.append(torch.nn.functional.linear(fine+residual[:,:,target],b.main_head.density.weight[rows,:,0],b.main_head.density.bias[rows]))
        manual=torch.stack(manual,2).reshape_as(active[0][1])
        affine_error=float((manual-active[0][1]).abs().max())
        assert torch.allclose(manual,active[0][1],atol=2e-6,rtol=1e-6)
        identity=b.target_scale.target_embedding.clone()
        b.target_scale.target_embedding[1].add_(.4)
        one_target=b(x,d,v,True,absolute_depth=d)
        b.target_scale.target_embedding.copy_(identity)
        assert torch.equal(one_target[0][0],active[0][0])
        assert torch.equal(one_target[0][1][:,:,[0,2]],active[0][1][:,:,[0,2]])
        assert (one_target[0][1][:,:,1]-active[0][1][:,:,1]).abs().max()>0
        assert all(torch.equal(q,w) for aa,bb in zip(active[1:],one_target[1:]) for q,w in zip(aa,bb))
        absolute_change=b(x,d,v,True,absolute_depth=torch.randn_like(d)*10000)
        assert all(torch.equal(q,w) for aa,bb in zip(active,absolute_change) for q,w in zip(aa,bb))
        assert torch.equal(original[0][0],active[0][0])
        assert all(torch.equal(q,w) for aa,bb in zip(original[1:],active[1:]) for q,w in zip(aa,bb))
        assert (original[0][1]-active[0][1]).abs().max()>0
        altered=x.clone();altered[~v]+=1000
        padded=b(altered,d,v,True,absolute_depth=d)
        assert all(torch.equal(q,w) for aa,bb in zip(active,padded) for q,w in zip(aa,bb))
        altered=x.clone();altered[1]+=100
        other=b(altered,d,v,True,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for aa,bb in zip(active,other) for q,w in zip(aa,bb))
    # Independently reconstruct masked alignment, linear/LN, query/key and3-way softmax.
    r=TargetScaleReadout().double();r.output.weight.data.normal_(0,.03)
    features=[torch.randn(2,c,l,dtype=torch.float64,requires_grad=True) for c,l in [(128,32),(96,16),(128,8)]]
    mask=torch.ones(2,32,dtype=torch.bool);mask[0,:7]=False
    result,weights=r.route(*features,mask)
    np_values=[]
    for i,(feature,proj,norm) in enumerate(zip(features,r.projections,r.norms)):
        source_mask=array(mask).reshape(2,-1,2**i).any(-1)
        aligned=interpolate_numpy(array(feature)*source_mask[:,None],32).transpose(0,2,1)
        projected=aligned@array(proj.weight).T
        normalized=(projected-projected.mean(-1,keepdims=True))/np.sqrt(projected.var(-1,keepdims=True)+norm.eps)
        np_values.append((normalized*array(norm.weight)+array(norm.bias))*array(mask)[:,:,None])
    nv=np.stack(np_values,2);keys=nv@array(r.key.weight).T
    queries=(array(features[0]).transpose(0,2,1)@array(r.query.weight).T)[:,:,None]+array(r.target_embedding)[None,None]
    scores=np.einsum('bltd,blsd->blts',queries.astype(np.float32),keys.astype(np.float32))/np.sqrt(32.)
    ww=np.exp(scores-scores.max(-1,keepdims=True));ww/=ww.sum(-1,keepdims=True);ww*=array(mask)[:,:,None,None]
    expected=np.einsum('blts,blsd->bltd',ww,nv)@array(r.output.weight).T
    formula_error=float(np.max(abs(expected-array(result))))
    assert formula_error<2e-6
    assert torch.allclose(weights.sum(-1),mask[:,:,None].float().expand(-1,-1,3),atol=1e-6)
    assert (weights[:,:,0]-weights[:,:,1]).abs().max()>1e-6
    with torch.no_grad():
        original_embedding=r.target_embedding.clone();permutation=torch.tensor([2,0,1])
        r.target_embedding.copy_(original_embedding[permutation]);permuted=r(*features,mask)
        assert torch.allclose(permuted,result[:,:,permutation],atol=1e-7)
        r.target_embedding.copy_(original_embedding)
        contribution=[]
        for i in range(3):
            altered=list(features);altered[i]=torch.zeros_like(features[i])
            effect=float((r(*altered,mask)-result).abs().max());assert effect>1e-6;contribution.append(effect)
        empty=r(*features,torch.zeros_like(mask));assert torch.equal(empty,torch.zeros_like(empty))
    result.square().mean().backward()
    assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in features)
    # Full-model loss keeps detached final point Q, masks and default continuous exclusions.
    b.train();outputs=b(x,d,v,True,absolute_depth=d);outputs[0][1].retain_grad()
    y=torch.randn(2,256,3);defaults=torch.rand(2,256,3)>.6
    supervised=v.clone();supervised[:,:64]=False;supervised[:,192:]=False
    q=torch.ones(2,256,requires_grad=True)
    loss=objective(outputs,y,defaults,supervised,True,q,epoch=20)
    scaled=objective(outputs,y,defaults,supervised,True,q*.2,epoch=20)
    assert torch.allclose(scaled,loss*.2,atol=1e-7)
    bad=y.clone();bad[~supervised]=1e5
    assert torch.equal(loss,objective(outputs,bad,defaults,supervised,True,q,epoch=20))
    loss.backward();assert q.grad is None
    assert (outputs[0][1].grad[defaults]==0).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in b.target_scale.parameters())
    assert all(p.device.type=='cpu' for p in b.parameters())
    out=Path('outputs/pure_dl_target_scale_readout_sequence');out.mkdir(exist_ok=True)
    summary={'passed':True,'scope':'CPU only; no GPU benchmark or training','parameters':682269,'added_parameters':20768,'same_old_parameters_and_RNG':True,'zero_residual_initial_identity':True,'gate_aux_unchanged_with_active_branch':True,'independent_numpy_routing_error':formula_error,'manual_density_affine_error':affine_error,'single_target_embedding_isolation':True,'absolute_depth_invariance':True,'target_identity_permutation':True,'three_scale_ablation_max_changes':contribution,'empty_padding_well_isolation':True,'all_three_scale_and_new_parameter_gradients':True,'Q_detached_and_final_loss_scaled':True,'default_continuous_gradient_zero':True,'strict_threshold':80.2}
    (out/'cpu_preflight.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

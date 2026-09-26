"""CPU-only GRL, training-domain ownership, loss and warmup-update checks."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline
from well_adversarial_sequence import Network,GradientReverse,grl_coefficient,pool_core,make_domain_mapping,domain_labels
from por_t3_patch_loss import objective
from fold5_screening_802 import self_test

def main():
    torch.set_num_threads(2);self_test();torch.manual_seed(75)
    feature=torch.randn(5,128,dtype=torch.float64,requires_grad=True)
    head=torch.nn.Linear(128,64).double();target=torch.tensor([0,1,2,3,63])
    standard=F.cross_entropy(head(feature),target)
    expected=torch.autograd.grad(standard,[feature,head.weight,head.bias])
    coefficient=.01
    reversed_input=GradientReverse.apply(feature,coefficient)
    assert torch.equal(feature,reversed_input)
    actual=torch.autograd.grad(F.cross_entropy(head(reversed_input),target),[feature,head.weight,head.bias])
    assert torch.equal(actual[0],-coefficient*expected[0])
    assert torch.equal(actual[1],expected[1]) and torch.equal(actual[2],expected[2])
    assert [grl_coefficient(i) for i in range(10)]==[0.]*10
    assert grl_coefficient(10)==.001 and grl_coefficient(19)==.01 and grl_coefficient(74)==.01
    wells=[i for i in range(80) if i%5!=4];mapping=make_domain_mapping(wells)
    assert torch.equal(domain_labels(torch.tensor(wells),mapping),torch.arange(64))
    rejected=False
    try:domain_labels(torch.tensor([4]),mapping)
    except AssertionError:rejected=True
    assert rejected and (mapping[torch.arange(4,80,5)]==-1).all()
    fine=torch.randn(3,128,256,requires_grad=True);valid=torch.ones(3,256,dtype=torch.bool)
    valid[0,64:70]=False;valid[2]=False
    pooled,usable=pool_core(fine,valid)
    for i in range(2):
        expected_mean=torch.stack([fine[i,:,j] for j in range(64,192) if valid[i,j]],1).mean(1)
        assert torch.allclose(pooled[i],expected_mean,atol=1e-7)
    assert usable.tolist()==[True,True,False] and (pooled[2]==0).all()
    pooled.sum().backward()
    mask=valid.clone();mask[:,:64]=False;mask[:,192:]=False
    assert (fine.grad.masked_select(~mask[:,None].expand_as(fine))==0).all()
    changed=fine.detach().clone();changed[:,:,192:]+=1e4;changed[:,:,:64]-=1e4;changed[0,:,64:70]+=1e4
    assert torch.equal(pool_core(changed,valid)[0],pooled)
    torch.manual_seed(91);a=Baseline().train();rng=torch.get_rng_state()
    torch.manual_seed(91);b=Network().train();assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in b.parameters())==669757
    x=torch.randn(2,256,206);depth=torch.zeros(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:19]=False
    y=torch.randn(2,256,3);defaults=torch.rand(2,256,3)>.5;use=v.clone();use[:,:64]=False;use[:,192:]=False
    q=torch.rand(2,256,requires_grad=True)
    ao=a(x,depth,v,True,absolute_depth=depth)
    bo,logits,usable=b(x,depth,v,True,absolute_depth=depth,return_domain=True,domain_coefficient=0.)
    assert usable.all() and all(torch.equal(t,u) for aa,bb in zip(ao,bo) for t,u in zip(aa,bb))
    original=objective(ao,y,defaults,use,True,q,epoch=0)
    prediction=objective(bo,y,defaults,use,True,q,epoch=0)
    assert torch.equal(original,prediction)
    assert torch.allclose(objective(bo,y,defaults,use,True,q*.2,epoch=0),prediction*.2,atol=1e-7)
    ap=list(a.parameters());bp=b.backbone_parameters();dp=list(b.domain_head.parameters())
    oa=torch.optim.AdamW(ap,lr=.0006,weight_decay=.01);ob=torch.optim.AdamW(bp,lr=.0006,weight_decay=.01);od=torch.optim.AdamW(dp,lr=.0006,weight_decay=.01)
    original.backward();(prediction+F.cross_entropy(logits,torch.tensor([0,1]))).backward()
    assert q.grad is None
    assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(ap,bp))
    torch.nn.utils.clip_grad_norm_(ap,2);torch.nn.utils.clip_grad_norm_(bp,2);torch.nn.utils.clip_grad_norm_(dp,2)
    oa.step();ob.step();od.step()
    warm_error=max(float((value-b.state_dict()[key]).abs().max()) for key,value in a.state_dict().items())
    assert warm_error==0.,warm_error
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in dp)
    b.eval()
    with torch.no_grad():
        before=b(x,depth,v,True,absolute_depth=depth)
        b.domain_head.weight.add_(100);b.domain_head.bias.add_(100)
        after=b(x,depth,v,True,absolute_depth=depth)
        assert all(torch.equal(t,u) for aa,bb in zip(before,after) for t,u in zip(aa,bb))
        changed=x.clone();changed[~v]+=100
        padded=b(changed,depth,v,True,absolute_depth=depth)
        assert all(torch.equal(t,u) for aa,bb in zip(before,padded) for t,u in zip(aa,bb))
        changed=x.clone();changed[1]+=100
        other=b(changed,depth,v,True,absolute_depth=depth)
        assert all(torch.equal(t[0],u[0]) for aa,bb in zip(before,other) for t,u in zip(aa,bb))
    b.train();b.domain_head.weight.data.div_(100);b.domain_head.bias.data.zero_();b.zero_grad(set_to_none=True)
    pred,dom,usable=b(x,depth,v,True,absolute_depth=depth,return_domain=True,domain_coefficient=.01)
    combined=objective(pred,y,defaults,use,True,torch.ones_like(q),epoch=20)+F.cross_entropy(dom,torch.tensor([0,1]))
    combined.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
    assert not b.combine._forward_hooks
    result={'passed':True,'scope':'CPU only; no GPU/training','parameters':669757,'added_parameters':8256,'GRL_backbone_exact_negative_lambda':True,'domain_head_normal_gradient':True,'train64_mapping_validation_rejected':True,'valid_core_pool_independent_formula_and_empty_mask':True,'pool_excludes_invalid_and_outside_core_gradients':True,'old_initialization_and_RNG_identical':True,'lambda0_CPU_AdamW_backbone_update_max_error':warm_error,'prediction_independent_of_domain_head':True,'padding_well_isolation':True,'original_prediction_Q_unchanged_and_detached':True,'combined_active_gradients_finite':True,'strict_threshold':80.2,'limitation':'CPU one-step exact equality does not guarantee GPU bitwise epoch trajectories; well identity can encode useful geology and lower classifier accuracy is not a success criterion'}
    out=Path('outputs/pure_dl_well_adversarial_sequence');out.mkdir(exist_ok=True);(out/'cpu_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

import json,torch
from pathlib import Path
from relative_only_sequence import Network,objective
from restore_trajectory_features import feature_contract

def main():
    torch.set_num_threads(2);torch.manual_seed(37);contract=feature_contract();assert contract['selected_feature_count']==216
    model=Network(features=216).eval();assert model.stem.weight.shape==(64,218,1)
    x=torch.randn(2,256,216);d=torch.randn(2,256);mask=torch.ones(2,256,dtype=torch.bool);mask[:,:30]=False
    with torch.no_grad():
        out=model(x,d,mask,absolute_depth=d)
        changed=x.clone();changed[1]+=100;second=model(changed,d,mask,absolute_depth=d)
        assert all(torch.equal(a[0],b[0]) for a,b in zip(out,second))
        changed=x.clone();changed[~mask]+=100;dd=d.clone();dd[~mask]+=100;second=model(changed,dd,mask,absolute_depth=dd)
        assert all(torch.equal(a,b) for a,b in zip(out,second))
        second=model(x,d,mask,absolute_depth=d+100)
        assert all(torch.equal(a,b) for a,b in zip(out,second))
    loss=objective(model(x,d,mask,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,mask,True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result={'passed':True,'features':216,'stem_channels':218,'restored':contract['restored'],'well_and_padding_isolation':True,'absolute_depth_ignored':True,'finite_all_gradients':True}
    out=Path('outputs/pure_dl_restore_trajectory');out.mkdir(parents=True,exist_ok=True);(out/'numerical_checks.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

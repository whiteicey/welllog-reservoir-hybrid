"""Per-depth feature-group attention, inspired by tabular Transformers.

Groups are fixed from input feature names, not selected from labels. This is
an adaptation with eleven curve/context tokens, not an FT-Transformer replica.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from train_invariant import feature_contract
from multiscale_sequence import DensityHead, objective


def feature_groups():
    names=feature_contract()['whitelist']
    raw={'GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT'}
    groups={key:[] for key in sorted(raw)};groups['DEPTH']=[];groups['CROSS']=[]
    for i,name in enumerate(names):
        hits=raw.intersection(name.split('_'))
        key=next(iter(hits)) if len(hits)==1 else ('DEPTH' if name=='DEPTH_REL' else 'CROSS')
        groups[key].append(i)
    assert sorted(i for values in groups.values() for i in values)==list(range(206))
    return groups


class Network(nn.Module):
    def __init__(self,features=206,initial_means=None):
        super().__init__();assert features==206
        groups=feature_groups();self.group_names=list(groups)
        self.projections=nn.ModuleList()
        for i,indices in enumerate(groups.values()):
            self.register_buffer(f'group_{i}',torch.tensor(indices,dtype=torch.long))
            self.projections.append(nn.Linear(len(indices),32))
        self.token_identity=nn.Parameter(torch.randn(1,12,32)*.02)
        self.cls=nn.Parameter(torch.zeros(1,1,32))
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(32,4,64,dropout=.1,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.point=nn.Sequential(nn.Linear(206,64),nn.GELU())
        self.combine=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1))
        self.main_head=DensityHead(128,initial_means)
        self.aux_mid=DensityHead(128,initial_means);self.aux_coarse=DensityHead(128,initial_means)

    def forward(self,features,relative_depth,valid,auxiliary=False):
        batch,length,_=features.shape
        x=(features*valid[:,:,None]).reshape(-1,206)
        tokens=torch.stack([layer(x[:,getattr(self,f'group_{i}')]) for i,layer in enumerate(self.projections)],1)
        tokens=torch.cat([self.cls.expand(len(x),-1,-1),tokens],1)+self.token_identity
        for layer in self.layers:tokens=layer(tokens)
        value=self.combine(torch.cat([tokens[:,0],tokens[:,1:].mean(1),self.point(x)],-1))
        value=value.reshape(batch,length,128).transpose(1,2)*valid[:,None]
        main=self.main_head(value)
        if not auxiliary:return main
        def pool(ratio):
            return F.avg_pool1d(value,ratio,ratio)/F.avg_pool1d(valid[:,None].float(),ratio,ratio).clamp_min(1e-6)
        return main,self.aux_mid(pool(2)),self.aux_coarse(pool(4))


def self_test():
    torch.set_num_threads(2);torch.manual_seed(139)
    model=Network().eval();x=torch.randn(2,16,206);depth=torch.randn(2,16)
    valid=torch.ones(2,16,dtype=torch.bool);valid[0,:3]=False
    with torch.no_grad():
        original=model(x,depth,valid)
        changed=x.clone();changed[~valid]+=1000
        assert all(torch.equal(a,b) for a,b in zip(original,model(changed,depth,valid)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(original,model(changed,depth,valid)))
        changed=x.clone();changed[0,8]+=100
        after=model(changed,depth,valid)
        assert all(torch.equal(a[0,:8],b[0,:8]) for a,b in zip(original,after))
    outputs=model(x,depth,valid,True)
    loss=objective(outputs,torch.randn(2,16,3),torch.rand(2,16,3)>.4,valid,True)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    return {'passed':True,'all206_features_partitioned_once':True,'padding_and_well_isolation':True,
            'main_attention_is_within_each_depth_row':True,'finite_attention_gradients':True,
            'parameters':sum(p.numel() for p in model.parameters()),'groups':{k:len(v) for k,v in feature_groups().items()}}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))

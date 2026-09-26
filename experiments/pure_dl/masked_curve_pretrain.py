"""Fold-local masked curve reconstruction before reservoir-label finetuning."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline, window_indices
from train_invariant import feature_contract

CURVES=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']


def curve_assignment():
    names=feature_contract()['whitelist'];ids={name:i for i,name in enumerate(CURVES)}
    assignments=[]
    for name in names:
        hits=set(name.split('_')).intersection(ids)
        assignments.append(ids[next(iter(hits))] if len(hits)==1 else (-2 if name=='DEPTH_REL' else -1))
    return assignments,[names.index(name) for name in CURVES]


def corrupt(features,masked_curves,assignments):
    safe=assignments.clamp_min(0)
    keep=~masked_curves[:,safe]
    keep[:,assignments==-1]=False  # Always hide cross-curve/unknown aliases in the pretext task.
    keep[:,assignments==-2]=True   # Relative depth is a legitimate reconstruction input.
    return features*keep[:,None].to(features.dtype)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);assert features==206
        assignments,targets=curve_assignment()
        self.register_buffer('curve_assignment',torch.tensor(assignments))
        self.register_buffer('reconstruction_indices',torch.tensor(targets))
        self.reconstruction=nn.Conv1d(128,9,1)

    def reconstruct(self,x,relative,valid):
        hidden=[]
        handle=self.combine.register_forward_hook(lambda module,args,value:hidden.append(value))
        try:super().forward(x,relative,valid)
        finally:handle.remove()
        return self.reconstruction(hidden[0]).transpose(1,2)


def pretrain(model,x,depth,lower,upper,training_raw,steps=600):
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    core=torch.zeros(1,256,device=x.device,dtype=torch.bool);core[:,64:192]=True
    history=[];model.train()
    for step in range(steps):
        anchors=training_raw[torch.randint(len(training_raw),(64,),device=x.device)]
        ix,exists=window_indices(anchors,lower,upper)
        raw=x[ix];relative=torch.tanh((depth[ix]-depth[anchors,None])/10)
        masked=torch.rand(64,9,device=x.device)<.35
        # Ensure at least one masked curve per window without consulting labels.
        empty=~masked.any(-1);masked[empty,step%9]=True
        corrupted=corrupt(raw,masked,model.curve_assignment)
        active=(exists&core)[:,:,None]&masked[:,None]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.reconstruct(corrupted,relative,exists)
        target=raw[:,:,model.reconstruction_indices].float()
        loss=F.smooth_l1_loss(pred.float()[active],target[active],beta=1.)
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step==0 or (step+1)%100==0:
            history.append({'step':step+1,'loss':float(loss.detach())})
            print('MASKED_PRETRAIN',history[-1],flush=True)
    opt.zero_grad(set_to_none=True)
    model.reconstruction.requires_grad_(False)
    return history


def self_test():
    torch.set_num_threads(2);torch.manual_seed(149)
    model=Network().eval();x=torch.randn(2,16,206);masked=torch.zeros(2,9,dtype=torch.bool);masked[:,0]=True
    corrupted=corrupt(x,masked,model.curve_assignment)
    assert (corrupted[:,:,model.curve_assignment==0]==0).all()
    assert (corrupted[:,:,model.curve_assignment==-1]==0).all()
    assert torch.equal(corrupted[:,:,model.curve_assignment==-2],x[:,:,model.curve_assignment==-2])
    altered=x.clone();altered[:,:,model.curve_assignment==0]+=1000
    assert torch.equal(corrupted,corrupt(altered,masked,model.curve_assignment))
    valid=torch.ones(2,16,dtype=torch.bool);relative=torch.randn(2,16)
    gate_before={k:v.clone() for k,v in model.main_head.state_dict().items()}
    prediction=model.reconstruct(corrupted,relative,valid)
    loss=F.smooth_l1_loss(prediction[:,:,0],x[:,:,model.reconstruction_indices[0]])
    loss.backward()
    assert model.stem.weight.grad.abs().sum()>0 and model.reconstruction.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.main_head.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(torch.equal(v,model.main_head.state_dict()[k]) for k,v in gate_before.items())
    return {'passed':True,'masked_curve_derived_columns_hidden':True,'cross_alias_columns_hidden':True,
            'masked_input_cannot_change_corrupted_input':True,'encoder_gets_pretext_gradients':True,
            'reservoir_heads_not_trained_by_pretext':True,'parameters':sum(p.numel() for p in model.parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))

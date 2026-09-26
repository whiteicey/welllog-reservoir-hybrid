import copy
import torch
class ParameterEMA:
    def __init__(self,model):
        self.model=copy.deepcopy(model).eval().requires_grad_(False);self.updates=0
    @torch.no_grad()
    def update(self,model):
        self.updates+=1;decay=min(.995,(1+self.updates)/(10+self.updates))
        target=list(self.model.parameters());source=list(model.parameters())
        assert len(target)==len(source)
        torch._foreach_mul_(target,decay);torch._foreach_add_(target,source,alpha=1-decay)
        for a,b in zip(self.model.buffers(),model.buffers()):a.copy_(b)
        return decay
if __name__=='__main__':
    model=torch.nn.Linear(2,1);average=ParameterEMA(model)
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    for step in range(1,21):
        with torch.no_grad():
            for p in model.parameters():p.add_(.1)
        decay=average.update(model)
        for k,v in model.state_dict().items():expected[k]=expected[k]*decay+v*(1-decay)
        assert all(torch.allclose(v,expected[k],atol=1e-6) for k,v in average.model.state_dict().items())
    assert all(not p.requires_grad for p in average.model.parameters())
    assert average.updates==20 and not average.model.training
    print('PASS EMA updates match independent recurrence; detached evaluation copy')

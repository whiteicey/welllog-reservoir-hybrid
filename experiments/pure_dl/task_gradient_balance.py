import torch
from population_likelihood_sequence import objective

def task_losses(outputs,z,default,mask,deep):
    result=[]
    for j in range(3):
        if deep:sliced=tuple((g[:,:,j:j+1],r[:,:,j:j+1]) for g,r in outputs)
        else:sliced=(outputs[0][:,:,j:j+1],outputs[1][:,:,j:j+1])
        result.append(objective(sliced,z[:,:,j:j+1],default[:,:,j:j+1],mask,deep))
    return torch.stack(result)
class Balancer:
    def __init__(self):self.ema=None;self.weights=None;self.step=0;self.trace=[]
    def combine(self,losses,shared):
        if self.step%16==0:
            norms=[]
            for loss in losses:
                grad=torch.autograd.grad(loss,shared,retain_graph=True)
                norms.append(torch.stack([g.detach().float().square().sum() for g in grad]).sum().sqrt())
            norms=torch.stack(norms).clamp_min(1e-8)
            self.ema=norms if self.ema is None else .9*self.ema+.1*norms
            self.weights=.5+1.5*(-self.ema.log()).softmax(0)
            assert torch.isfinite(self.weights).all()
            self.trace.append({'step':self.step,'norms':norms.cpu().tolist(),'ema':self.ema.cpu().tolist(),'weights':self.weights.cpu().tolist()})
        self.step+=1
        return (losses*self.weights.detach()).mean()
    def __call__(self,outputs,z,default,mask,deep,shared):
        return self.combine(task_losses(outputs,z,default,mask,deep),shared)
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(281)
    gate=torch.randn(2,8,3,requires_grad=True);raw=torch.randn(2,8,3,3,8,requires_grad=True);z=torch.randn(2,8,3);d=torch.rand(2,8,3)>.5;m=torch.ones(2,8,dtype=torch.bool)
    assert torch.allclose(task_losses((gate,raw),z,d,m,False).mean(),objective((gate,raw),z,d,m,False),atol=1e-6)
    parameter=torch.nn.Parameter(torch.ones(3));losses=parameter*torch.tensor([1.,4.,10.]);bal=Balancer();value=bal.combine(losses,[parameter])
    assert bal.weights[0]>bal.weights[1]>bal.weights[2]
    assert abs(float(bal.weights.sum())-3)<1e-6 and (bal.weights>=.5).all() and (bal.weights<=2).all()
    value.backward();assert torch.allclose(parameter.grad,torch.tensor([1.,4.,10.])*bal.weights/3)
    print('PASS equal-task decomposition, bounded mean-one weights, detached weighted gradients')

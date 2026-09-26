import torch
from torch import nn
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class SpectralBlock(nn.Module):
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(torch.randn(192,8,2)*.02)
        self.project=nn.Conv1d(192,192,1);self.gain=nn.Parameter(torch.full((1,192,1),.1))
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            spectrum=torch.fft.rfft(x.float(),dim=-1,norm='ortho')
            filtered=torch.zeros_like(spectrum);k=min(8,spectrum.shape[-1])
            filtered[:,:,:k]=spectrum[:,:,:k]*torch.view_as_complex(self.weights.contiguous())[:,:k]
            value=torch.fft.irfft(filtered,n=x.shape[-1],dim=-1,norm='ortho')
            value=self.project(value)
        return x+self.gain*value.to(x.dtype)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);self.bottleneck.add_module('spectral',SpectralBlock())
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(251);model=Network(features=12)
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    outputs=model(x,d,v,True,absolute_depth=a)
    loss=objective(outputs,torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert model.bottleneck.spectral.weights.grad.abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    model.eval()
    with torch.no_grad():
        first=model(x,d,v,absolute_depth=a);changed=x.clone();changed[1]+=100
        assert all(torch.equal(u[0],w[0]) for u,w in zip(first,model(changed,d,v,absolute_depth=a)))
    print('PASS spectral finite complex gradients and well isolation')

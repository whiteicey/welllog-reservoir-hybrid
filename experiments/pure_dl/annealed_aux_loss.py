"""Let coarse training objectives retire while keeping final point weighting intact."""
from global_point_loss import head_loss

def objective(outputs, target, default, mask, deep_supervision=False, weights=None, *, epoch):
    assert weights is not None and weights.shape == mask.shape
    if not deep_supervision:
        return head_loss(outputs,target,default,mask,weights)
    main,mid,coarse=outputs
    factor=max(0.,1.-epoch/40.)
    loss=head_loss(main,target,default,mask,weights)
    if factor:
        loss=loss+factor*(.2*head_loss(mid,target,default,mask,weights)+.1*head_loss(coarse,target,default,mask,weights))
    return loss

if __name__=='__main__':
    import torch
    from global_point_loss import objective as original
    torch.set_num_threads(2);torch.manual_seed(1)
    outputs=tuple((torch.randn(2,n,3,requires_grad=True),torch.randn(2,n,3,3,8,requires_grad=True)) for n in [16,8,4])
    y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.7;m=torch.ones(2,16,dtype=torch.bool);m[:,:2]=False
    q=torch.rand(2,16)*.9+.1
    first=objective(outputs,y,d,m,True,q,epoch=0)
    assert torch.allclose(first,original(outputs,y,d,m,True,q),atol=1e-6)
    last=objective(outputs,y,d,m,True,q,epoch=40)
    assert torch.equal(last,head_loss(outputs[0],y,d,m,q))
    middle=objective(outputs,y,d,m,True,q,epoch=20)
    assert torch.allclose(middle,(first+last)/2)
    last.backward()
    assert all(x.grad is None for pair in outputs[1:] for x in pair)
    assert outputs[0][1].grad[~m].abs().sum()==0
    assert torch.isfinite(outputs[0][1].grad).all()
    print('PASS: epoch0 baseline loss, linear schedule, epoch40 main-only gradient, masking')

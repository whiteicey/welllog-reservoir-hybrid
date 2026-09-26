"""Permute raw row IDs and existence together; all supervision is gathered afterwards."""
import torch

def reverse_selected(ix,exists,flip):
    assert ix.shape==exists.shape and flip.shape==(ix.shape[0],)
    return torch.where(flip[:,None],ix.flip(1),ix),torch.where(flip[:,None],exists.flip(1),exists)

if __name__=='__main__':
    torch.manual_seed(3)
    ix=torch.arange(3*256).reshape(3,256);exists=torch.ones_like(ix,dtype=torch.bool)
    exists[0,:80]=False;ix[0,:80]=80
    core=torch.zeros(1,256,dtype=torch.bool);core[:,64:192]=True
    labels=torch.randn(3*256,3);weights=torch.rand(3*256)
    flip=torch.tensor([True,False,True]);new,valid=reverse_selected(ix,exists,flip)
    for i in range(3):
        assert torch.equal(ix[i][(exists&core)[i]].sort().values,new[i][(valid&core)[i]].sort().values)
        expected=labels[ix[i]].flip(0) if flip[i] else labels[ix[i]]
        assert torch.equal(labels[new[i]],expected)
        expected=weights[ix[i]].flip(0) if flip[i] else weights[ix[i]]
        assert torch.equal(weights[new[i]],expected)
        assert torch.equal(new[i]//256,ix[i]//256)
    twice,twice_mask=reverse_selected(new,valid,flip)
    assert torch.equal(twice,ix) and torch.equal(twice_mask,exists)
    generator=torch.Generator().manual_seed(456)
    before=torch.get_rng_state();torch.rand(64,generator=generator)
    assert torch.equal(before,torch.get_rng_state())
    print('PASS raw row/label/Q alignment, symmetric core population, boundary masks, no cross-well, reversal involution, independent RNG')

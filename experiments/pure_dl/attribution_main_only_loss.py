"""Remove only auxiliary prediction losses; original main likelihood retained."""
from por_t3_patch_loss import head_loss


def objective(outputs,target,default,mask,deep_supervision=False,weights=None,*,epoch):
    assert len(outputs)==2 and deep_supervision
    # Keep network/output interfaces and initialization unchanged. Auxiliary
    # heads are still evaluated, but neither auxiliary loss enters gradients.
    return sum(head_loss(member[0],target,default,mask,weights) for member in outputs)/2

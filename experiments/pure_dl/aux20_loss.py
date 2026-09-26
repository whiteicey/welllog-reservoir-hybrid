"""Original POR-t3 objective with only the linear auxiliary fade shortened to20."""
from por_t3_patch_loss import head_loss, probability, point_statistics


def objective(outputs, target, default, mask, deep_supervision=False, weights=None, *, epoch):
    assert weights is not None and weights.shape == mask.shape
    if not deep_supervision:
        return head_loss(outputs, target, default, mask, weights)
    main, mid, coarse = outputs
    factor = max(0., 1. - epoch / 20.)
    loss = head_loss(main, target, default, mask, weights)
    if factor:
        loss = loss + factor * (
            .2 * head_loss(mid, target, default, mask, weights)
            + .1 * head_loss(coarse, target, default, mask, weights))
    return loss

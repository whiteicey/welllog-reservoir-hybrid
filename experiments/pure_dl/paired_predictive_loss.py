"""Optimize the actual equal full predictive mixture, not mean member NLL."""
from paired_backbone_sequence import merge_predictions
from por_t3_patch_loss import objective as distribution_objective


def objective(outputs, target, default, mask, deep_supervision=False, weights=None, *, epoch):
    assert len(outputs) == 2 and deep_supervision
    merged = tuple(merge_predictions([outputs[0][i], outputs[1][i]]) for i in range(3))
    return distribution_objective(merged, target, default, mask, True, weights, epoch=epoch)

"""Matched role-swap control for complementary target priorities."""
from paired_target_specialist_loss import member_objective


TARGET_WEIGHTS = ((.75, .75, 1.5), (1.25, 1.25, .5))


def paired_objective(outputs, target, default, mask, deep_supervision, weights, *, epoch):
    assert deep_supervision and len(outputs) == 2
    return sum(member_objective(output, target, default, mask, weights, TARGET_WEIGHTS[index], epoch=epoch)
               for index, output in enumerate(outputs)) / 2

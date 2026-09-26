"""Fixed first-fold screening; equality with82.3 does not qualify."""
import math

THRESHOLD = 82.3


def qualifies(score):
    return math.isfinite(float(score)) and float(score) > THRESHOLD


def self_test():
    assert not qualifies(82.3)
    assert not qualifies(81.7976691057382)
    assert not qualifies(float('nan'))
    assert not qualifies(float('inf'))
    assert qualifies(82.300001)
    return {'passed': True, 'strict_threshold': THRESHOLD, 'nonfinite_scores_rejected': True}

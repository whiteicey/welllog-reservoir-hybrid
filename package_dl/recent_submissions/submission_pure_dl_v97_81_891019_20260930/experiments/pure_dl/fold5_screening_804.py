"""User-selected fifth-fold screening; original GroupKFold index4, strict80.4."""
import math
THRESHOLD=80.4
FOLD_INDEX=4

def qualifies(score):return math.isfinite(float(score)) and float(score)>THRESHOLD

def fold_plan(count):
    assert count in (1,5)
    return [FOLD_INDEX] if count==1 else [FOLD_INDEX,0,1,2,3]

def self_test():
    assert not qualifies(80.4) and qualifies(80.400001)
    assert not qualifies(float('nan')) and not qualifies(float('inf'))
    assert fold_plan(1)==[4] and sorted(fold_plan(5))==list(range(5))
    return {'passed':True,'screening_fold':5,'strict_threshold':THRESHOLD}

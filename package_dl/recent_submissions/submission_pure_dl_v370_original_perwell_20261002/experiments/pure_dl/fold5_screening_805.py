"""Latest user threshold: original grouped fifth fold must strictly exceed80.5."""
import math

THRESHOLD=80.5
FOLD_INDEX=4


def qualifies(score):
    return math.isfinite(float(score)) and float(score)>THRESHOLD


def fold_plan(count):
    assert count in (1,5)
    return [FOLD_INDEX] if count==1 else [FOLD_INDEX,0,1,2,3]

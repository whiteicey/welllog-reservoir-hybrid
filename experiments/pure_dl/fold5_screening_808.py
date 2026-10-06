"""Latest explicit original-fold5 promotion gate; historical helpers immutable."""
THRESHOLD=80.8
FOLD_INDEX=4
def qualifies(value):return float(value)>THRESHOLD

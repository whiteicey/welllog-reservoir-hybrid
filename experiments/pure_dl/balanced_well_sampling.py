"""One uniformly sampled complete-label anchor from every outer-training well."""
import torch


def sample_anchors(raw_rows,offsets,lengths):
    assert offsets.shape==lengths.shape and bool((lengths>0).all())
    local=(torch.rand(len(lengths),device=raw_rows.device)*lengths).long()
    return raw_rows[offsets+local]


def self_test():
    torch.manual_seed(159)
    rows=torch.tensor([2,4,6,20,21,100,105,110,115])
    offsets=torch.tensor([0,3,5]);lengths=torch.tensor([3,2,4])
    observed=[set() for _ in range(3)]
    for _ in range(100):
        anchors=sample_anchors(rows,offsets,lengths)
        assert len(anchors)==3
        for i,value in enumerate(anchors.tolist()):
            assert value in rows[offsets[i]:offsets[i]+lengths[i]].tolist()
            observed[i].add(value)
    assert [len(s) for s in observed]==lengths.tolist()
    return {'passed':True,'one_anchor_from_each_training_well':True,'never_selects_outside_supplied_training_rows':True,'all_rows_remain_eligible':True}

"""Matched private PERM backbone with input-only50% row/50% well anchor sampling."""
import hashlib
import torch
from perm_private_attention_v173 import ROOT,SOURCE,PARENT,STAGES,Network,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_well_sampling_v175_20261001'
OUT=ROOT/'outputs/pure_dl_perm_well_sampling_v175_fold5'
def make_model(stage,master,metadata):return Network(master,stage)
def prepare_sampling(data):
    anchors=data['train_raw'];well=data['row_well'][anchors]
    order=well.argsort(stable=True);sorted_anchors=anchors[order];ids,counts=torch.unique_consecutive(well[order],return_counts=True)
    assert len(ids)==64 and int(counts.sum())==len(anchors)
    starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
    fractions=counts.double()/counts.sum();mixture=.5*fractions+.5/len(ids)
    manifest=dict(mode='50percent row-uniform+50percent well-uniform',well_ids=ids.cpu().tolist(),training_anchor_counts=counts.cpu().tolist(),well_sampling_probabilities=mixture.cpu().tolist(),labels_used=False,validation_used=False,model_well_ID_input=False,training_anchors=len(anchors),sorted_training_anchors_sha256=hashlib.sha256(sorted_anchors.cpu().numpy().tobytes()).hexdigest(),same_seed_not_same_sample_sequence=True)
    return dict(anchors=anchors,sorted_anchors=sorted_anchors,counts=counts,starts=starts,manifest=manifest)
def sample_anchors(sampling,generator,batch=64):
    anchors=sampling['anchors'];row=anchors[torch.randint(len(anchors),(batch,),device='cuda',generator=generator)]
    wells=torch.randint(len(sampling['counts']),(batch,),device='cuda',generator=generator)
    offset=(torch.rand(batch,device='cuda',generator=generator)*sampling['counts'][wells]).long()
    balanced=sampling['sorted_anchors'][sampling['starts'][wells]+offset]
    choose=torch.rand(batch,device='cuda',generator=generator)<.5
    return torch.where(choose,balanced,row)

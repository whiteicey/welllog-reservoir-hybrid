"""Retain measured well-sampling gain, matched whole-model SAM optimizer study."""
from perm_well_sampling_v175 import ROOT,SOURCE,PARENT,Network,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_well_sam_v177_20261001'
OUT=ROOT/'outputs/pure_dl_perm_well_sam_v177_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,6e-4)}
def make_model(stage,master,metadata):return Network(master,stage)

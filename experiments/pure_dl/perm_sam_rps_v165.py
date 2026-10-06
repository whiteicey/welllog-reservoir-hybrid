"""Matched PERM AdamW/SAM0.05 study on positive RPS density readout."""
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_sam_rps_v165_20261001'
OUT=ROOT/'outputs/pure_dl_perm_sam_rps_v165_fold5'
STAGES={'adamw':('PERM',256,False,0.,3e-4),'sam':('PERM',256,True,0.,3e-4)}
def make_model(stage,master,metadata):return Network(master,stage)

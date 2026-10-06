"""Carry measured SAM benefit into the PERM default-gate update scope."""
from perm_gate_coadapt_v161 import ROOT,SOURCE,PARENT,Network,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_sam_gate_rps_v168_20261001'
OUT=ROOT/'outputs/pure_dl_perm_sam_gate_rps_v168_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
def make_model(stage,master,metadata):return Network(master,stage)

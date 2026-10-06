"""Transfer the measured well-sampling mechanism into positive PERM conditional readout."""
from perm_conditional_cdf_v174 import ROOT,SOURCE,PARENT,STAGES,Network,load_data,Evaluator,channel_objective,make_model
from perm_well_sampling_v175 import prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_head_well_sampling_v180_20261001'
OUT=ROOT/'outputs/pure_dl_perm_head_well_sampling_v180_fold5'

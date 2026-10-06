"""Same local SW architecture; tensor-RMS weighted SAME-L2-radius SAM direction."""
from compact_frequency_v104 import ROOT,SOURCE,Network as Parent,load_data
STUDY=ROOT/'results/deep_learning/compact_sw_scaled_sam_v123_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_scaled_sam_v123_fold5'
STAGES={'rms_direction':('SW',1024,True,1.,3e-4)}
class Network(Parent):
    def __init__(self,metadata):super().__init__(metadata,'local')
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(sam_direction='tensor RMS plus0.01 weighted gradient, re-normalized to same globalL2rho0.1',
            sam_floor=.01,sam_rho=.1,same_network_init_and_capacity=True,metric='same Euclidean L2 perturbation norm, different direction',
            not_standard_ASAM=True,no_radius_or_LR_sweep=True,original_score_and_decoder=True);return info
def make_model(stage,master,metadata):return Network(metadata)

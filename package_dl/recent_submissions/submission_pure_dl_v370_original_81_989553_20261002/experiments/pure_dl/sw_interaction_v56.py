"""Matched additive versus multiplicative feed-forward blocks in private SW."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from divergent_models_v45 import Network as RandomNetwork
from well_conditioned_v46 import ROOT,Network as Conditioned,WellEncoder,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_interaction_v56_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_sharpness_v55_20260929'
SOURCE=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('additive','multiplicative')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class InteractionFF(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.input=nn.Linear(96,512);self.norm=nn.LayerNorm(256);self.output=nn.Linear(256,96)
    def forward(self,x):
        a,b=self.input(x).chunk(2,-1)
        z=(F.silu(a)+b)/math.sqrt(2) if self.mode=='additive' else F.silu(a)*b
        return self.output(self.norm(z))

class Network(RandomNetwork):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('long_section',metadata);self.c=2;self.mode='well';self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261146);self.condition=WellEncoder()
        self.register_buffer('reference_inputs',metadata['reference_inputs'].detach().clone(),persistent=False)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261156)
            for block in self.body.global_blocks:block.ff=InteractionFF(mode)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.initialization_info=dict(source=dict(checkpoint=str(SOURCE.relative_to(ROOT)),
            sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_best_epoch=saved['best_epoch'],
            source_target='SW',source_training_population='same original64trainingwells'),
            backbone='SW-trained attention/TCN/FiLM; four FF blocks replaced by identically initialized new parameters',
            reset_full_output_head=True,own_private_parameters=True,optimizer_reset=True,
            current_supervision='SW only',new_ff=dict(blocks=4,branch_width=256,input=96,normalization='shared post-combination LayerNorm',
                additive='(SiLU(a)+b)/sqrt(2)',multiplicative='SiLU(a)*b',same_parameter_values_across_modes=True))
    def initialize_priors(self,data):
        prior=super().initialize_priors(data)
        prior['scope']='only64trainingwell labels/quality fit output priors; backbone partly transferred and four FF blocks new'
        return prior|self.initialization_info
    forward=Conditioned.forward

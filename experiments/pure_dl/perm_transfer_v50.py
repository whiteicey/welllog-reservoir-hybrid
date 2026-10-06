"""Private PERM network: random backbone versus own-fold SW representation transfer."""
import hashlib
import numpy as np
import torch
from torch import nn
from divergent_models_v45 import Network as RandomNetwork
from well_conditioned_v46 import ROOT,Network as Conditioned,WellEncoder,Evaluator as SWNamedEvaluator,load_data

STUDY=ROOT/'results/deep_learning/perm_transfer_v50_20260929'
PREVIOUS=ROOT/'results/deep_learning/quantile_supervision_v49_20260929'
SOURCE=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('scratch','sw_transfer')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(RandomNetwork):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('long_section',metadata)
        self.c=1;self.mode='well';self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261146);self.condition=WellEncoder()
        self.register_buffer('reference_inputs',metadata['reference_inputs'].detach().clone(),persistent=False)
        source=None
        if mode=='sw_transfer':
            saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
            self.load_state_dict(saved['state_dict'],strict=True)
            source=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                source_target='SW',source_training_population='same original64trainingwells',source_best_epoch=saved['best_epoch'])
        # No source gate/density prediction is transferred or ensembled.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.initialization_info=dict(backbone='random' if mode=='scratch' else 'SW-trained own-fold body/norm/FiLM',
            source=source,reset_full_output_head=True,own_private_parameters=True,optimizer_reset=True,
            current_supervision='PERM only; cross-target labels used only in historical source training')

    def initialize_priors(self,data):
        return super().initialize_priors(data)|self.initialization_info

    forward=Conditioned.forward

class Evaluator(SWNamedEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        prediction,_,scores=super().__call__(model)
        return prediction,float(scores['PERM']),scores

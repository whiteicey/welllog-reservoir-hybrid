"""Four-stage private SW with frozen stage-two POR semantic input features."""
import json
import numpy as np
import torch
from torch import nn
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_frequency_v104 import Network as LocalNetwork

STUDY=ROOT/'results/deep_learning/compact_por_features_v108_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_features_v108_fold5'
TEACHER=ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por/model_best.pt'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('curve_control','por_posterior')}

class Network(LocalNetwork):
    def __init__(self,metadata):
        super().__init__(metadata,'local')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261108);self.body.stem=nn.Linear(211,96)

def load_data(stage):
    data,master,metadata=original_data('sw_joint')
    manifest=json.loads((STUDY/'feature_manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['rows']==len(data['x'])
    features=np.load(STUDY/(stage+'_features.npy'))
    assert features.shape==(len(data['x']),4) and np.isfinite(features).all()
    data['x']=torch.cat((data['x'],torch.tensor(features,device='cuda')),-1)
    metadata['semantic_feature_manifest']=manifest
    return data,master,metadata

def make_model(stage,master,metadata):return Network(metadata)

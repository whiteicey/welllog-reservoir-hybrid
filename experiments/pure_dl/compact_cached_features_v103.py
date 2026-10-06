"""Prepared alternative: native 256-point source context cached once per row.

This changes the source context definition relative to v102; it is not claimed
to reproduce moving SW-anchor source features. No trial starts automatically.
"""
import json
import numpy as np
import torch
from torch import nn
from compact_five_stage_v100 import ROOT,SOURCE,CompactSW,load_data as original_data

STUDY=ROOT/'results/deep_learning/compact_cached_features_v103_20261001'
OUT=ROOT/'outputs/pure_dl_compact_cached_features_v103_fold5'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('random','pretrained')}

class Network(CompactSW):
    def __init__(self,metadata):
        super().__init__(metadata)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261102);self.body.stem=nn.Linear(463,96)

def load_data(stage):
    data,master,metadata=original_data('sw_joint')
    manifest=json.loads((STUDY/(stage+'_cache.json')).read_text(encoding='utf-8'))
    assert manifest['rows']==len(data['x']) and manifest['source_predictions_used'] is False
    assert manifest['complete'] and manifest['source_sha256']==__import__('hashlib').sha256(SOURCE.read_bytes()).hexdigest()
    array=np.load(STUDY/(stage+'_features.npy'),mmap_mode='r')
    assert array.shape==(len(data['x']),256) and array.dtype==np.float16
    # The well bank was built from original raw206 inputs before concatenation.
    latent=torch.tensor(np.array(array),device='cuda');data['x']=torch.cat((data['x'],latent),-1)
    metadata['feature_cache_manifest']=manifest
    return data,master,metadata

def make_model(stage,master,metadata):return Network(metadata)

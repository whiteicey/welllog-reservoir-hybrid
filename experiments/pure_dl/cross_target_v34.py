"""SW refinement with independent, group-excluded auxiliary predictions.
Self/cross/shuffled have exactly the same trainable architecture and initialization.
The immutable champion supplies its own SW hidden states and distribution.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crosscurve_v32 import Network as FrozenBuilder,factor_scope,channel_objective
from champion_all_heads_v26 import Evaluator,load_data as parent_data,TARGETS

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/cross_target_v34_20260929'
PREVIOUS=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
MODES=('self','cross','shuffled')
LEARNING_RATE=3e-4

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def load_data():
    data,master=parent_data()
    manifest=json.loads((STUDY/'feature_manifest.json').read_text(encoding='utf-8'))
    assert manifest['auxiliary_query_wells_excluded_from_fit_and_stopping'] and not manifest['original_fifth_labels_used']
    assert sorted(np.unique(data['wells'][data['fold']!=4]).tolist())==sorted(manifest['fit_wells'])
    assert data['x'].shape==(730268,269)
    with np.load(STUDY/'feature_views.npz') as z:views=[z[m] for m in MODES]
    assert all(v.shape==(730268,12) and np.isfinite(v).all() for v in views)
    data['x']=torch.cat((data['x'],*[torch.as_tensor(v,device='cuda',dtype=data['x'].dtype) for v in views]),-1)
    assert data['x'].shape==(730268,305)
    return data,master

class Correction(nn.Module):
    def __init__(self,target,mode):
        super().__init__()
        if target!='SW' or mode not in MODES:raise ValueError((target,mode))
        self.target=target;self.mode=mode;self.factor='weight'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261135)
            self.input=nn.Conv1d(331,32,1)
            self.output=nn.Conv1d(32,16,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,hidden,parent,context,side,valid):
        if side.shape[-1]!=99:raise ValueError('Expected amplitude63 + three12 auxiliary views')
        k=MODES.index(self.mode);view=side[...,63+12*k:75+12*k]
        z=torch.cat((hidden[0],hidden[1],side[...,:63].transpose(1,2).to(hidden[0].dtype),view.transpose(1,2).to(hidden[0].dtype)),1)
        z=z.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            proposal=self.output(F.gelu(self.input(z))).transpose(1,2).float()
        proposal=proposal.masked_fill(~valid[...,None],0)
        gate,raw=parent
        assert raw.shape[-3:]==(1,3,16)
        delta=.25*torch.tanh(proposal/.25)
        return gate,torch.stack((raw[...,0,:]+delta.unsqueeze(2),raw[...,1,:],raw[...,2,:]),-2)

class Network(FrozenBuilder):
    def __init__(self,target,master,modes=MODES):
        if target!='SW':raise ValueError('This study only refines SW')
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:Correction(target,m) for m in modes})
        self.metadata.update(feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)),
            feature_manifest_sha256=hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest(),
            auxiliary_manifest=str((STUDY/'auxiliary_manifest.json').relative_to(ROOT)),
            modes=list(MODES),head_input=331,head_hidden=32,head_output=16,
            source='small private cross-fitted auxiliary models; no raw other-target labels at SW head',
            negative_control='shuffle matching within each entire excluded group; prevents cross-group source leakage',
            primary_parent_no_crossfit_claim=True)
        self.train(False)

def mechanism_probe(model,data,mode):
    result=factor_scope(model,data,mode)
    assert result['max_bound_ratio']==0
    result.update(zero_output_initialization=True,parent_original269_only=True,mode=mode,
        auxiliary_values=12,head_input=331,old_SW_gate_and_locations_and_scales_fixed=True)
    return result

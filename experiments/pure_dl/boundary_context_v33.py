"""Fixed51 versus input-boundary-truncated51 side statistics on frozen v31.

Identical small correction heads; only 18 side statistics differ. No label
segmentation, no output smoothing, and no changes to the parent input path.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from crosscurve_v32 import Network as PreviousFeatureNetwork,Correction as PreviousCorrection
from crosscurve_v32 import factor_scope,channel_objective
from champion_all_heads_v26 import Evaluator,load_data as parent_data,TARGETS

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/boundary_context_v33_20260929'
PREVIOUS=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
MODES=('fixed','segmented')
LEARNING_RATE=3e-4
FACTORS={'POR':'location','PERM':'gate','SW':'weight'}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def load_data():
    data,master=parent_data()
    meta=json.loads((STUDY/'feature_manifest.json').read_text(encoding='utf-8'))
    assert not meta['labels_used_to_construct_features'] and not meta['cross_well_windows']
    assert sorted(np.unique(data['wells'][data['fold']!=4]).tolist())==sorted(meta['fit_wells'])
    assert data['x'].shape==(730268,269)
    with np.load(STUDY/'feature_views.npz') as z:views=[z[m] for m in MODES]
    assert all(v.shape==(730268,18) and np.isfinite(v).all() for v in views)
    data['x']=torch.cat((data['x'],*[torch.as_tensor(v,device='cuda',dtype=data['x'].dtype) for v in views]),-1)
    return data,master

class Correction(PreviousCorrection):
    def __init__(self,target,mode):
        if mode not in MODES:raise ValueError(mode)
        # Reuse the identical tested readout and initialization. Internal names
        # only select the first/second18 slots; no v32 feature values are loaded.
        super().__init__(target,('marginal','crosscorr')[MODES.index(mode)])
        self.boundary_view=mode

class Network(PreviousFeatureNetwork):
    def __init__(self,target,master,modes=MODES):
        # Reuse the v31 parent builder and immutable current checkpoints.
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:Correction(target,m) for m in modes})
        self.metadata.update(feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)),
            feature_manifest_sha256=hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest(),
            views=list(MODES),implementation_reuse='crosscurve_v32 frozen parent and identical small readout; no v32 learned weights',
            labels_used_for_boundaries=False,control='ordinary centered51 means/stds',
            treatment='same51 maximum window clipped at nearest input peaks')
        self.train(False)

def mechanism_probe(model,data,mode):
    result=factor_scope(model,data,mode)
    assert result['max_bound_ratio']==0.
    result.update(zero_output_initialization=True,parent_original269_only=True,feature_view=mode,
        additional_view_dimensions=18,input_projection_dimensions=337,labels_used_for_features=False)
    return result

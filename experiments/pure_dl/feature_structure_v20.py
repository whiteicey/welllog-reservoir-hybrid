"""Equal-parameter feature/late-fusion study with a private POR distribution head."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from por_architecture_v18 import Network as PORBase
from por_data_v18 import load_data as load_por_data
from independent_sw_scratch_v10 import TemporalBlock
from architecture_zoo_v15 import clean

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/feature_structure_v20_20260928'
MODES=('amplitude_early','shape_early','shape_late')

def load_feature_data(mode):
    if mode not in MODES:raise ValueError(mode)
    d=load_por_data()
    key='amplitude' if mode=='amplitude_early' else 'shape'
    with np.load(STUDY/'side_views.npz') as z:
        side=torch.from_numpy(z[key]).cuda()
    assert side.shape==(d['x'].shape[0],63) and bool(torch.isfinite(side).all())
    d['x']=torch.cat((d['x'],side),-1)
    return d

class Network(PORBase):
    def __init__(self,mode,metadata):
        nn.Module.__init__(self)
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.width=80;w=self.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            self.stem=nn.Linear(207,w)
            self.side_stem=nn.Linear(63,w)
            self.blocks=nn.ModuleDict({
                'main':nn.ModuleList([TemporalBlock(w,1),TemporalBlock(w,2)]),
                'side':nn.ModuleList([TemporalBlock(w,1),TemporalBlock(w,2)]),
                'gate':nn.Linear(2*w,w),
                'decoder':nn.ModuleList([TemporalBlock(w,4),TemporalBlock(w,8)])})
            self.norm=nn.LayerNorm(w)
            self.output=nn.Linear(w,25)
        nn.init.zeros_(self.blocks['gate'].weight);nn.init.zeros_(self.blocks['gate'].bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('centers',torch.as_tensor(metadata['centers'],dtype=torch.float32).clone())
        self.register_buffer('density_edges',torch.as_tensor(metadata['density_edges'],dtype=torch.float32).clone())
        self.target_median=float(metadata['target_median']);self.target_scale=float(metadata['target_scale'])
        self.metadata=metadata
        with torch.no_grad():
            self.output.bias[0].copy_(torch.logit(torch.tensor(float(metadata['training_default_rate']))))
            self.output.bias[9:17].copy_(torch.as_tensor(metadata['initial_means'],dtype=torch.float32))

    def encoded_views(self,x,relative,valid):
        if x.shape[-1]!=269:raise ValueError('Expected original206 + side63 input columns')
        main=clean(torch.cat((x[...,:206].float(),relative[...,None].float()),-1),valid)
        side=clean(x[...,206:].float(),valid)
        a=clean(F.gelu(self.stem(main)),valid)
        b=clean(F.gelu(self.side_stem(side)),valid)
        if self.mode!='shape_late':
            a=b=clean((a+b)*.5,valid)
        for block in self.blocks['main']:a=block(a,valid)
        for block in self.blocks['side']:b=block(b,valid)
        gate=self.blocks['gate'](torch.cat((a,b),-1)).sigmoid()
        return a,b,gate

    def forward(self,x,relative,valid):
        a,b,gate=self.encoded_views(x,relative,valid)
        h=clean(gate*a+(1-gate)*b,valid)
        for block in self.blocks['decoder']:h=block(h,valid)
        output=clean(self.output(self.norm(h)).float(),valid)
        return output[...,0],output[...,1:].reshape(*output.shape[:2],3,8)

    @torch.no_grad()
    def mechanism_statistics(self):
        return dict(mode=self.mode,width=self.width,side_features=63,
            view='amplitude' if self.mode=='amplitude_early' else 'local_standardized_shape',
            separate_view_encoding=self.mode=='shape_late',
            content_fusion_gate_weight_norm=float(self.blocks['gate'].weight.float().norm()),
            side_stem_weight_norm=float(self.side_stem.weight.float().norm()),
            head='8xStudentT_df3+default_gate',all_parameters_trainable=True)

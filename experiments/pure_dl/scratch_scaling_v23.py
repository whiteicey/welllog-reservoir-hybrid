"""Random-network private POR dual experts; controlled internal-width growth.

No old network state is loaded. The old train-only input/target metadata and
Q50 are deliberately retained. This is single-target predictive-likelihood
training, not a replay of the historical multi-target auxiliary pretraining.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from independent_channels_v8 import Network as PrivateNetwork, TargetReadout, TargetRouterOutput
from paired_local_context_router_sequence import Network as Original, retained_indices, calibration_indices
from dual8222_location_readout_v5 import BoundedLocationResidual
from blur_ungated_sequence import UngatedBlock
from champion_scaling_v21 import ExpandedBlock, ExpandedLocation, load_data, REFERENCE

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/scratch_scaling_v23_20260928'
MODES={'base1':1,'ffn2':2}


class RandomReadout(TargetReadout):
    """Keep v8 packed arithmetic, with only random POR rows as parameters."""
    def __init__(self, old):
        nn.Module.__init__(self)
        self.target=0; self.valid=None
        for prefix, layer, stop in (('gate',old.gate,1),('density',old.density,24)):
            for suffix in ('weight','bias'):
                name=prefix+'_'+suffix
                tensor=getattr(layer,suffix).detach()
                self.register_parameter(name,nn.Parameter(tensor[:stop].clone()))
                self.register_buffer(name+'_before',tensor[:0].clone())
                self.register_buffer(name+'_after',tensor[stop:].clone())
        self.residual=BoundedLocationResidual(local=False)


def signature(state, exclude_extra=False):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if exclude_extra and '.extra_' in name: continue
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Network(PrivateNetwork):
    def __init__(self, metadata, mode):
        nn.Module.__init__(self)
        if mode not in MODES: raise ValueError(mode)
        self.target='POR';self.target_index=0;self.architecture='dual'
        self.mode=mode;self.factor=MODES[mode]
        names=metadata['selected_feature_names']
        numeric,condition=calibration_indices(names)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            # Other output rows are temporary random construction buffers only;
            # their initialization repeats POR means and reads no other label.
            means=np.tile(np.asarray(metadata['initial_means_POR']), (3,1))
            original=Original(initial_means=means,retain_indices=retained_indices(names),
                numeric_indices=numeric,conditioning_indices=condition)
            members=[]
            for member in original.members:
                member.main_head=RandomReadout(member.main_head)
                member.aux_mid=nn.Identity();member.aux_coarse=nn.Identity()
                members.append(member)
            self.members=nn.ModuleList(members)
            self.router=original.router
            self.router[2]=TargetRouterOutput(self.router[2],0)
            self.register_buffer('router_indices',original.router_indices.detach().clone())
            del original
        base_count=sum(p.numel() for p in self.parameters())
        assert base_count==1290490, base_count
        common_before=signature(self.state_dict())
        blocks=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261031)
            for prefix,parent in list(self.named_modules()):
                for name,child in list(parent.named_children()):
                    if isinstance(child,UngatedBlock):
                        blocks.append(dict(path=prefix+'.'+name,width=child.expand.in_channels))
                        if self.factor>1:setattr(parent,name,ExpandedBlock(child,self.factor))
            assert len(blocks)==18
            if self.factor>1:
                for member in self.members:
                    member.main_head.residual=ExpandedLocation(member.main_head.residual,self.factor)
        assert signature(self.state_dict(),exclude_extra=True)==common_before
        self.scaling_manifest=dict(mode=mode,internal_width_factor=self.factor,
            parameters=sum(p.numel() for p in self.parameters()),
            extra_parameters=sum(p.numel() for n,p in self.named_parameters() if '.extra_' in n),
            base_parameters=base_count,blocks=blocks,location_hidden_width=32*self.factor,
            common_initial_state_sha256=common_before,initial_state_sha256=signature(self.state_dict()),
            source_network_weights_loaded=False,all_network_parameters_trainable=True,
            external_channels_attention_and_router_fixed_in_design=True)
        assert self.scaling_manifest['parameters']=={'base1':1290490,'ffn2':1981498}[mode]


def metadata_from_training(data,master):
    labels=data['labels'][data['train_raw'],0].detach().cpu().numpy()
    default=data['default'][data['train_raw'],0].detach().cpu().numpy()
    means=np.quantile(labels[~default],np.linspace(.05,.95,8))
    return dict(selected_feature_names=list(master['selected_feature_names']),
        initial_means_POR=means.tolist(),initial_means_source='only64 training wells, POR nondefault standardized labels',
        network_initialization_seed=20261010,expansion_seed=20261031,
        source_network_weights_loaded=False)
